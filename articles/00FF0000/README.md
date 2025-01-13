# Training an *(almost)* Unbeatable Tic-Tac-Toe AI using Reinforcement Learning

Hello! If you are reading this, you probably have read the
associated [article](https://www.neuralnova.net/blog/rl-for-tic-tac-toe/introduction/)
and are looking to replicate the results.
This sub-folder contains all the code I reference in the article,
enabling you to follow along and train your own version of the tic-tac-toe agents.

* If you just want the final ONNX model files, you can find them under `00FF0000/exports`.
* If you would like to use the raw RlLib training results without running training on your
own machine, you will need to download the zipped results from the blog. 
The results folder is quite messy for git to keep track of, so I decided to exclude it
from the repository.

My results for the top 5 trials were as follows:

| Trial    | Metric | Iters | Best Iter | Learning Rate | Tie Penalty | Gamma  |
|----------|--------|-------|-----------|---------------|-------------|--------|
| d0b80b71 | 0.9735 | 241   | 236       | 0.000107      | -0.2389     | 0.9225 |
| 4f927e2f | 0.9551 | 242   | 236       | 0.000445      | 0.6819      | 0.8238 |
| 1bdac927 | 0.9382 | 243   | 243       | 0.000256      | 0.4762      | 0.9685 |
| 2d565d0a | 0.8453 | 241   | 227       | 0.001006      | 0.2669      | 0.9662 |
| 5dc51969 | 0.8108 | 243   | 243       | 0.000392      | 0.7170      | 0.9690 |

I used `checkpoint_000011` from trial `d0b8ob71` to create the ONNX files that are 
located in `00FF0000/exports`