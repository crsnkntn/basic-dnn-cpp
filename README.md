# Basic Deep Neural Network Written in C++

## What this is good for:
This dnn is good for creating basic single decision models. 

## Why this is important:
Coding this was a useful experience that demystified what deep neural nets are.

## How to use this:
The user creates an input token, output token, and then writes an activation function. Then they create a derived dnn class that implements functions to enter the tokens into the input of the net and extract the result of the feed forward algorithm. Currently this is only able to be used for making single predictions, so for training an integer is entered as the correct answer for back propogation. 