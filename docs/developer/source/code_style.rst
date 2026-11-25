Code style
==========

How PyRS code, tests and docstrings are written. The short form of these rules is in ``CLAUDE.md`` (the
instructions for Claude Code); this page has the full examples.

Functions
---------

Every public function or class has:

1. **Type hints** on parameters and return values.
2. **A Google-style docstring**: ``Args:``, ``Returns:``, ``Raises:``, and ``Example:`` where it helps. Use this
   style in new code even in modules whose older code uses NumPy-style docstrings (``Parameters`` / ``----------``).
3. **Specific exceptions** with clear messages that name the offending value (``f"... got {value}"``).
4. **Input validation** at boundaries.
5. **Comments only when the why is non-obvious**: never narrate what the code does.

.. code-block:: python

    def example_function(
        data: list[float],
        threshold: float = 0.5,
        normalize: bool = True,
    ) -> list[float]:
        """
        Process data with filtering and optional normalization.

        Args:
            data: Raw measurement values from sensor.
            threshold: Minimum value to keep (default: 0.5).
            normalize: Whether to normalize to [0, 1] range (default: True).

        Returns:
            Processed data values.

        Raises:
            ValueError: If data is empty or threshold is negative.

        Example:
            >>> example_function([0.1, 0.7, 1.2], threshold=0.5)
            [0.5833333333333334, 1.0]
        """
        if not data:
            raise ValueError("Data cannot be empty")
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")

        filtered = [x for x in data if x >= threshold]
        if not filtered:
            return []

        if normalize:
            max_val = max(filtered)
            return [x / max_val for x in filtered]
        return filtered

Tests
-----

Every test follows Arrange-Act-Assert, with ``# Arrange``, ``# Act`` and ``# Assert`` comments. Test naming, the
test tiers and their markers are in ``CLAUDE.md`` ("Testing Guidelines") and ``tests/CLAUDE.md``.

.. code-block:: python

    def test_filters_below_threshold():
        """Values below the threshold are removed."""
        # Arrange
        input_data = [0.1, 0.5, 0.7, 1.0]
        threshold = 0.6

        # Act
        result = example_function(input_data, threshold=threshold)

        # Assert
        assert len(result) == 2
        assert all(x >= threshold for x in result)

Modules and classes
-------------------

A module docstring says what the module is for and its typical workflow:

.. code-block:: python

    """
    Module for data preprocessing operations.

    Provides functions for cleaning and normalizing experimental data from
    the XYZ instrument. Typical workflow:

    1. Load raw data with load_data().
    2. Clean with remove_outliers().
    3. Normalize with normalize_values().
    4. Export with save_processed_data().
    """

A class docstring describes its purpose, its key attributes and a usage example.
