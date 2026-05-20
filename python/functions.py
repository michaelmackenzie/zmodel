import zfit

# Define a custom PowerLaw PDF class
class PowerLaw(zfit.pdf.ZPDF):
    """
    Custom 1D Power-Law PDF: f(x) = x^(gamma)
    For a falling spectrum like DIO, gamma will optimize to a negative number.
    """
    # Define the parameter names that this PDF depends on
    _PARAMS = ("gamma",)

    @zfit.supports(norm=False)
    def _pdf(self, x, norm, params):
        # Extract the coordinate tensor (axis 0)
        data = x[0]
        gamma = params["gamma"]

        # Return the unnormalized mathematical definition
        return znp.power(data, gamma)

