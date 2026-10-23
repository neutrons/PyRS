Using `Config` from `neutrons_standard`:
----------------------------------------

Using This Package
Add PythonCommons as a dependency to your project:

```yaml
# In your pyproject.toml
dependencies = [
    "PythonCommons @ git+https://github.com/neutrons/PythonCommons.git",
]
```
or

```ini
[tool.pixi.workspace]
channels = [
  "neutrons",
  "conda-forge",
  "https://prefix.dev/pixi-build-backends",
]

[tool.pixi.dependencies]
neutrons_standard = "*"

[tool.pixi.package.run-dependencies]
neutrons_standard = "*"
```

Then import utilities:

```python
from neutrons_standard import Config
from neutrons_standard.decorators.singleton import Singleton
from neutrons_standard.time import timestamp, isoFromTimestamp
```
