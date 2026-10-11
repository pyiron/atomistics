# Atomistics
The `atomistics` package provides "interfaces for  atomistic simulation codes and workflows". While the 
[user documentation](https://atomistics.readthedocs.io/) already covers how to use the `atomistics` package, this 
document explains the motivation and vision of the development the `atomistics` package, the underlying concepts and 
the resulting internal structure of the `atomistics` package. 

## Vision 
The `atomistics` package provides "interfaces for  atomistic simulation codes and workflows".

To calculate a physical property with an atomistic simulation code, there are typically two ways: 

* **internal implementation**: The simulation code already provides the internal functionality to calculate the physical
  property. Most atomistic simulation codes can calculate, potential energies and forces for a given atomistic 
  structure. Some simulation codes already provide internal functionality to calculate more complex physical properties, 
  like the phonon density of states or energy barriers using the nudged elastic band method. 
* **external tools**: The second approach is to use a workflow, which combines a series of individual calculation of 
  atomistic simulation codes to calculate physical properties. These workflows can be simple shell scripts for example 
  for the calculation of energy volume curves or more complex physical properties, like the phonon density introduced 
  above. 

The challenge for the `atomistics` package is balancing both approaches. The first is commonly computationally more 
efficient, while the second allows to directly compare different simulation codes and simulation methods. The 
`atomistics` packages provides an abstraction to balance between both approaches. By default, it uses the internal 
implementation when available and falls back to a universal python based implementation when necessary. With this 
approach the `atomistics` package allows users to calculate a wide-range of physical properties with a wide-range of 
simulation codes. 

## Concepts
To address the challenge of balancing both the internal implementations and external tools, the `atomistics` package
defines a range of interfaces. These interfaces are a series of functions, which follow the same pattern.

### Functional Approach
The use of functions is preferred over the use of classes as applying and developing functions is typically easier to 
for scientists compared to working with classes. Classes are only used when transferring data between function calls. 

### Dictionary based Interfaces
Each function accessible to the users should return a dictionary to name the output of the function. 

### Selective Output
Each function defines the `output` parameter, which is a tuple of names of quantities the function calculates:

```python
def function(*args, **kwargs, output=('key_1', 'key_2', ...)):
    ...
    return {
        'key_1': ...,
        'key_2': ...,
    }
```

Based on this tuple each function returns a dictionary, where the keys are given by the names of the quantities defined
in the output parameter and the values are the results calculated from the function. This allows the user to select 
which output is calculated. 

## Structure 
The `atomistics` package is structured into the three following modules. 

### `atomistics/calculators`
Interfaces for the internal implementations of the individual simulation codes to calculate a specific physical property. 

### `atomistics/shared`
Classes, functions and modules which are used for both the internal interfaces and the external tools. 

### `atomistics/workflows`
Interfaces for external tools to calculate a specific physical property by combining physical properties calculated from
individual simulation codes.
