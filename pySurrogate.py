import numpy as np

# ============= Classes ================

class DefaultScaler:
  """ A placeholder scaler that mapped to the identity transformation to be used
  for creating custom sklearn-like scalers.

  ### Parameters
  fit_transform : {function}
    A function that fits the scaler to the data X and then transforms them.
  inverse_transform : {function}
    A function that scales the data back to their original representation.
  transform : {function}
    A function that transforms the data to the scaler's representation.
  """
  def __init__(self, fit_transform = lambda x: x, inverse_transform = lambda y: y, transform = lambda xy: xy):
    self.fit_transform = fit_transform
    self.inverse_transform = inverse_transform
    self.transform = transform


class Initializer:
  """ Takes care of input X transformations in different representations to match
  the input units and shape of the Surrogate model.

  ### Parameters
  inputTranformer : {function}
    A function that transforms the input vector to the desired shape or size.
  xScaler : {sklearn-like}, default = DefaultScaler().
    An sklearn-like scaler for X data with access to fit_transform, inverse_transform, transform.
  yScaler : {sklearn-like}, default = DefaultScaler().
    An sklearn-like scaler for Y data with access to fit_transform, inverse_transform, transform.
  """
  def __init__(self, inputTranformer, xScaler = DefaultScaler(), yScaler = DefaultScaler()):
    self.inputTranformer = inputTranformer
    self.xScaler = xScaler
    self.yScaler = yScaler
  
  def fitScalerXY(self, X: np.array, Y: np.array) -> np.array:
    """ Calls the fit_transform functions of the provided (if any) scalers to train
    them on the data and returns scaled (if at all) data.

    ### Parameters
    X : {np.array}, default = None
      The X array used to train xScaler.
    Y : {np.array}, default = None
      The Y array used to train yScaler.
    
    ### Returns
    X_tr : {np.array}, default = None
      The scaled X array.
    Y_tr : {np.array}, default = None
      The scaled Y array.
    """
    X_tr = self.xScaler.fit_transform(X)
    Y_tr = self.yScaler.fit_transform(Y)
    
    return X_tr, Y_tr
  
  def scale(self, X: np.array = None, Y: np.array = None) -> np.array:
    """ Scales X, Y (or both) input through the previously fit scalers.

    ### Parameters
    X : {np.array}, default = None
      The X array used to scale along the features axis of xScaler.
    Y : {np.array}, default = None
      The Y array used to scale along the features axis of yScaler.
    
    ### Returns
    X_tr : {np.array}, default = None
      The scaled X array.
    Y_tr : {np.array}, default = None
      The scaled Y array.
    """
    if X is not None and Y is not None:
      return self.xScaler.transform(X), self.yScaler.transform(Y)
    elif X is not None:
      return self.xScaler.transform(X)
    elif Y is not None:
      return self.yScaler.transform(Y)
  
  def descale(self, X: np.array = None, Y: np.array = None) -> np.array:
    """ Reverts the X, Y (or both) input data to their original representation.

    ### Parameters
    X : {np.array}, default = None
      The X array used to revert using the xScaler.
    Y : {np.array}, default = None
      The Y array used to revert using the yScaler.
    
    ### Returns
    X_tr : {np.array}, default = None
      The descaled X array.
    Y_tr : {np.array}, default = None
      The descaled Y array.
    """
    if X is not None and Y is not None:
      return self.xScaler.inverse_transform(X), self.yScaler.inverse_transform(Y)
    elif X is not None:
      return self.xScaler.inverse_transform(X)
    elif Y is not None:
      return self.yScaler.inverse_transform(Y)


class Decoder:
  """ The Decoder class provides uniformity and easy access to reconstructing
  data from a latent space representation and/or applying (input-based)
  transformations.

  This class is initialized with the decomposed coefficients on the latent space
  and the inverse transformation function which decodes the coefficients back to
  their original representation.

  If a basis is provided, the inverse transformation is assumed to be a linear
  projection by default.
  
  When called runs decodeCoefficients.

  ### Parameters
  projection_coeffs : {np.array}, default = np.array([])
    The X array used to scale along the features axis of xScaler.
  inverse : {function}, default = lambda x: x
    A function used to decode the latent space representation coefficients back to
    their original representation.
  basis : {np.array}, default = None
    The projection basis used to reconstruct the projection_coeffs as an alternative
    way from providing "inverse".
  """

  def __init__(self, projection_coeffs: np.array = np.array([]), inverse = lambda x: x, basis = None):
    self.projection_coeffs = projection_coeffs
    self.basis = basis
    self.inverse = inverse

    # When basis is provided, overwrite the inversion method.
    if basis is not None:
      self.inverse = lambda coeffs: coeffs @self.basis
      
    self.physicalTransform = lambda x, input: x

  def getSize(self):
    """ Returns the size of the latent space representation. """
    
    return len(self.projection_coeffs.T)
  
  def decodeCoefficients(self, coeffs: np.array, input: np.array) -> np.array:
    """ Reconstructs the latent space representation back to the original representation
    through inversion and application of the physical transformation.
    
    ### Parameters
    coeffs : {np.array}
      The coefficients of the latent space representation to be decoded.
    input : {np.array}
      The physical parameters of the model which are related to the latent space coefficients.
    
    ### Returns
    Y_decoded : {np.array}
      The reconstructed data in its original representation and dimensionality.
    """
    
    decoded = self.inverse(coeffs)
    Y_decoded = self.physicalTransform(self.inverse(coeffs), input)
    
    return Y_decoded
  
  def __call__(self, coeffs, input):
    return self.decodeCoefficients(coeffs, input)
    

class Regressor:
  """ A wrapper class for whichever regressor is used for the predicting mechanism
  of the Surrogate that provides calling uniformity.

  This class is simply initialized by providing the predicting model itself and
  the method it uses to predict.

  ### Parameters
  model : {object}
    A predictive model that has already been trained on some data.
  predict : {function}
    The method used by the predictive model to produce predictions.
  """

  def __init__(self, model, predict):
    self.model = model
    self.predict = predict
  
  def __call__(self, x: np.array) -> np.array:
    return self.predict(x)


class Surrogate:
  """ A wrapper class for easily producing predictive surrogate models of multi-dimensional
  data.

  This class is initialized with 3 parts:
    * The Initializer which controls input X representation transformations.
    * The Decoder which controls latent space reconstruction and physical transformations
    of Y data.
    * The Regressor which wrapps the predictive model which was previously trained
    on X-Y data.

  When called runs getPrediction.
  
  ### Parameters
  initializer : {Initializer}
    An Initializer object which controls transformation of input data X.
  decoder : {Decoder}
    A Decoder object with a trained decoder.
  regressor : {Regressor}
    A Regressor object with a trained predictive model.
  """

  def __init__(self, initializer: Initializer, decoder: Decoder, regressor: Regressor):
    self.initializer = initializer
    self.decoder = decoder
    self.regressor = regressor

    self.predict_raw = lambda x: self.initializer.descale(Y = self.regressor(x))
    self.predict = lambda x, input: self.decoder(self.predict_raw(x), input)
  
  def getPrediction(self, input: np.array, predictLatent: bool = False) -> np.array:
    """ Returns the prediction of the surrogate model on the input data.
    
    ### Parameters
    input : {np.array}
      The physical parameters to calculate prediction on.
    predictLatent : {np.array}, default = False
      When set to True, the Surrogate model will return its predictions on the latent
      space without decoding it back to the original representation.
    
    ### Returns
    Y_pred : {np.array}
      The predicted output of the Surrogate model.
    """
    # Transform and scale input data accordingly.
    X = self.initializer.inputTranformer(input)
    X = self.initializer.scale(X)

    return self.predict(X, input) if not predictLatent else self.regressor(X)
  
  def __call__(self, input: np.array, predictLatent = False) -> np.array:
    return self.getPrediction(input, predictLatent)