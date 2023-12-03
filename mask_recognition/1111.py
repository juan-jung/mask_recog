import inspect
import cvlib as cv

src = inspect.getsource(cv.detect_face)
print(src)