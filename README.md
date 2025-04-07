# Einops Implementation

## Overview
This module implements a subset of the functionality provided by the [einops](https://github.com/arogozhnikov/einops) library. The purpose of the module is to allow tensor manipulation operations such as reshaping, transposition, splitting, merging, and repeating axes using an Einstein notation-inspired pattern string. This module supports [Numpy](https://numpy.org/doc/) arrays. 

## Key Features
- **Reshaping and Transposition:** Transform the shape of NumPy arrays according to specified input and output patterns.
- **Splitting and Merging Axes:** Supports splitting of a single axis into multiple axes and merging of multiple axes into one.
- **Repeating Axes:** Allows the introduction of new axes with repeated values.
- **Ellipsis Handling:** Supports the use of ellipsis (`...`) to represent batch dimensions or arbitrary numbers of dimensions.
- **Robust Error Checking:** Provides informative error messages for invalid pattern strings, mismatched tensor shapes, and missing or extra axes lengths.

## Design Decisions
1. **Pattern Parsing:** The `ParsedExpression` class is responsible for parsing the pattern string. It processes the input into a composition of axes, recognizing ellipsis, grouped axes (for splitting/merging), and anonymous axes.
2. **Axes Resolution:** The `resolve_axes_lengths` function infers the sizes of the axes based on the input tensor shape and any additional length arguments provided.
3. **Transformation Steps:** The `rearrange` function follows a multi-step approach:
   - **Expansion:** The input pattern is expanded, and the tensor is reshaped accordingly.
   - **Insertion of New Axes:** Any new axes specified only in the output pattern are added via expansion and repetition.
   - **Transposition:** The axes of the tensor are transposed to match the order specified in the output pattern.
   - **Final Reshape:** The tensor is reshaped to account for any grouped axes in the output.
4. **Error Handling:** The implementation raises custom `EinopsError` exceptions with detailed error messages to help diagnose issues with input patterns or tensor shapes.

## Functions & Class Description: 
### `EinopsError` Class:
- EinopsError is a custom exception class used to raise clear, descriptive errors when an invalid pattern or operation occurs in the rearrange logic.

### `AnonymousAxis` Class:
- It represents unnamed axes defined by a positive integer (Example: 2 in a pattern)
- It is used for splitting or expanding dimensions without giving them a name.

### `ParsedExpression` Class:
- It parses a rearrangement pattern string (like 'a (b c) -> a b c') into a structured format.
- It validates axis names, grouping, and handling ellipsis.

### `resolve_axes_lengths` Function:
- It determines the size of each axis in the pattern by matching the parsed input pattern to the tensor's actual shape
- It handles ellipses, grouped axes, and inferred dimensions.

### `rearrange` function:
- It transforms a tensor's shape based on a pattern by parsing the input/output axes, resolving dimensions, reshaping, transposing.
- Finally applies a new shape as specified by the pattern.


## How to Run
#### Easiest way to use this is to download the implementation notebook or import it into gogglecolab. 

#### Or you can run it in your system following below steps:
### Prerequisites

- Python 3.7 or above
- NumPy
- Pytest (for running the tests)

### Installation
Clone the repository and navigate to the project directory


