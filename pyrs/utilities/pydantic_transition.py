from pydantic import ConfigDict, validate_call

# Use Pydantic's `validate_call` decorator with arbitrary types:
#   name with trailing-underscore to avoid conflicts if `pydantic.validate_call`
#   is itself ever used anywhere in the code-base.
def validate_call_(func):
    return validate_call(config=ConfigDict(arbitrary_types_allowed=True))(func)
