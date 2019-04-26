# Instrument Geometry

## Mantid

## PyRS-instrument

It is assumed that for HB2B,

1. sample position is always at (0, 0, 0)
2. source (moderator) position is always at (0, 0, -???)
3. detector can be configured to
  - 1024, 1024
  - 2048, 2048
4. detector pixels' size can be defined flexibly.

# Reduction Workflow

## Mantid

## PyRS-reduction

This is a reduction algorithm based on pure python programming.
It is supposed to be equivalent to the reduction Mantid algorithms,
which are practically slower.