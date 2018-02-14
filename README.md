# Text-indenpendent voice vectors

## Subtitle: which of the Hollywood stars is most similar to my voice?
>* Work in progress.
>* Authors: Dabi Ahn(andabi412@gmail.com), [Noah Jung](https://github.com/giallo41), and [Kyubyong Park](https://github.com/Kyubyong)

## Prologue
Everyone on earth has their own voice.
The same voice will not exist from different people, but some people have similar voices while others do not.
This project aims to find individual voice vectors using [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb) dataset, which contains 1,251 Hollywood stars' 145,379 utterances.
The voice vectors are text-independent, meaning that any pair of utterances from same speaker has similar voice vectors.
Also the closer the vector distance is, the more similar voices should appear.

<p align="center"><img src="https://raw.githubusercontent.com/andabi/voice-vector/master/materials/title.png" width="75%"></p>

## Architectures
The architecture is based on a classification model.
The utterance inputted is classified as one of the Hollywood stars.
The objective function is simply a cross entropy between speaker labels of ground truth and prediction.
Eventually, the last layer's activation becomes the speaker's embedding.

<p align="center"><img src="https://raw.githubusercontent.com/andabi/voice-vector/master/materials/arch.png" width="100%"></p>

The model architecture is structured as follows.
1. memory cell
    * the utterances inputted are randomly selected and cropped so that it does not matter to text.
    * CBHG module from [Tacotron](https://arxiv.org/abs/1703.10135) captures hidden features from sequential data such as speech utterances.
2. embedding
    * memory cell's last output is projected by the size of embedding vector.
3. softmax
    * embedding is logits for each classes.

## Datasets
* voxceleb description
  * meta desc.

* common voice desc.
  * meta desc.
  * age, gender dist.

## Results
* train/eval accuracy
* t-SNE (gender, age)

## How to run?
* normal mode
* remote mode

## The lessons

## Future works
* Triplet loss

## References
* Nagrani, A., Chung, J. S., & Zisserman, A. (2017, June 27). [VoxCeleb: a large-scale speaker identification dataset](http://arxiv.org/abs/1706.08612v1). arXiv.org.
* Zhang, C., & Koishida, K. (2017). [End-to-End Text-Independent Speaker Verification with Triplet Loss on Short Utterances](http://www.isca-speech.org/archive/Interspeech_2017/abstracts/1608.html) (pp. 1487–1491). Presented at the Interspeech 2017, ISCA: ISCA. http://doi.org/10.21437/Interspeech.2017-1608
* Li, C., Ma, X., Jiang, B., Li, X., Zhang, X., Liu, X., et al. (2017, May 6). [Deep Speaker: an End-to-End Neural Speaker Embedding System](http://arxiv.org/abs/1705.02304v1). arXiv.org.