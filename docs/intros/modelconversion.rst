Converting a scikit-learn model to ONNX
=======================================

This tutorial will demonstrate how you can use the Array API and ndonnx to convert a machine learning model to ONNX.

We will use scikit-learn to train a simple logistic regression model to classify species of the Iris plant using the classic `Iris Plant dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset>`_ and use ndonnx to convert it to ONNX.
We'll begin with a few imports and by loading the dataset.

..  code-block:: python

        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        import numpy as np
        import ndonnx as ndx
        from sklearn.model_selection import train_test_split
        import onnx

        dataset = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)


Extend predict in an Array API compatible manner
------------------------------------------------

We should begin by noting that only the inference logic is relevant for deployment. Since we would like to fit our classification model using scikit-learn’s ``LogisticRegression`` estimator, we are interested in exporting the logic of it's prediction method to ONNX.

Unfortunately, ``LogisticRegression.predict`` does not yet adhere to the Array API which would have made it immediately convertible to ONNX using ndonnx. scikit-learn’s Array API support is growing over time - functionality that is already Array API compliant will be ONNX convertible without extra work thanks to ndonnx.

Fortunately, we can easily extend LogisticRegression with our own class ``LogisticRegressionArrayAPI``. This extends the prediction logic in an Array API compliant fashion for Array API compliant inputs.

..  code-block:: python

    class LogisticRegressionArrayAPI(LogisticRegression):
        def predict(self, X):
            if hasattr(X, "__array_namespace__"):
                xp = X.__array_namespace__()
                coef = xp.asarray(self.coef_)
                intercept = xp.asarray(self.intercept_)
                classes = xp.asarray(self.classes_)
                index = xp.argmax(X @ coef.T + intercept, axis=1)
                return xp.take(classes, xp.astype(index, xp.int32), axis=0)
            else:
                return super().predict(X)

We can now fit our extended LogisticRegression model.

..  code-block:: python

    model = LogisticRegressionArrayAPI(C=1e5)
    model.fit(X_train, y_train)

Notice how versatile our implementation is now. We can make predictions using a variety of array backends like NumPy, JAX and ndonnx.

..  code-block:: python

    import ndonnx as ndx

    print(model.predict(X_test))
    # array([0, 0, 0, 1, 2, 2, 0, 1, 2, 1, 0, 2, 1, 1, 1, 2, 2, 0, 0, 1, 0, 1, 1, 0, 0, 2, 2, 2, 2, 2])

    print(model.predict(ndx.asarray(X_test)).to_numpy())
    # array([0, 0, 0, 1, 2, 2, 0, 1, 2, 1, 0, 2, 1, 1, 1, 2, 2, 0, 0, 1, 0, 1, 1, 0, 0, 2, 2, 2, 2, 2])


Exporting to ONNX
-----------------

Now that we have fit ``model`` and it’s inference path ``predict`` is Array API compatible, we can export it to ONNX using ndonnx.

1. Begin by creating a placeholder array representing the input of our model. These are arrays that contain no data, only a shape (which may be symbolic) and data type. The shape is (“N”, 4) since we may have an arbitrary batch dimension “N” and always have 4 features as input.

    ..  code-block:: python

        X = ndx.argument(shape=("N", 4), dtype=ndx.float64)

2. Call ``predict`` just as normal, providing X as input. The output array also does not have any data associated with it since its value depends on ``X``.

    ..  code-block:: python

        y = model.predict(X)
        assert y.to_numpy() is None

3. Build the ONNX graph with :func:`ndonnx.build` and persist it to disk. The dictionary names are the names given to the inputs and outputs in the ONNX graph.

    ..  code-block:: python

        onnx_model = ndx.build({"X": X}, {"y": y})
        onnx.save(onnx_model, "classify_iris.onnx")

4. Visualize the ONNX model using Netron. It’s a fairly small model and you might be able to carefully map some of the ONNX operators to the operations that generated them.

    .. image:: ../_static/classify_iris.png
        :alt: Iris Classification Model
        :align: center

Use the ONNX model in production
--------------------------------

Now we have our model computation and weights saved to disk as an ONNX file, we have tremendous amounts of flexibility to integrate our model into a wider system.
We can use an ONNX backend like onnxruntime to run our model. Here we use onnxruntime's python bindings to make a prediction.

..  code-block:: python

    import onnxruntime as ort

    # Instantiate runtime session
    inference_session = ort.InferenceSession("classify_iris.onnx")

    # Inference!
    out, = inference_session.run(None, {"X": X_test})

    print(out)
    # array([0, 0, 0, 1, 2, 2, 0, 1, 2, 1, 0, 2, 1, 1, 1, 2, 2, 0, 0, 1, 0, 1, 1, 0, 0, 2, 2, 2, 2, 2])
