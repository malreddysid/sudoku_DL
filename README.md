## Sodoku Solver using Deep Learning

The learning based method given in [this blog](http://aishack.in/tutorials/sudoku-grabber-opencv-plot/) and the non-learning based method that I tried [here](https://github.com/malreddysid/sudoku) were not very accurate. This project uses the LeNet Convolutional Neural Network for digit recognition to extract the digits from the grid of the sudoku puzzle.

## How to Setup

This project uses the Caffe framework to run the CNN.

1. Download the latest Caffe version [here](https://github.com/BVLC/caffe).
2. Copy the contents of `models/` folder into the `caffe/models/` folder.
3. Copy the contents of `examples/` folder into the `caffe/examples/` folder.
4. Build Caffe according to the instructions given on their website.
5. From the `caffe/` folder, run the following command
```shell
./build/examples/sudoku/sudoku.bin
```

## Running for different images

The image file is set in `caffe/examples/sudoku/sudoku.cpp` in line number `570`.

Edit the line, build caffe again and run the command.

## Implementation

You can read more about the implementation in my blog [here](https://malreddysid.github.io/deep_learning/2016/07/23/sudoku-solver-dl.html).

## TODO

* Improve the accuracy by finetuning with more data.
