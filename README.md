# Text-independent voice vectors

## Subtitle: which of the Hollywood stars is most similar to my voice?
>* Authors: Dabi Ahn(andabi412@gmail.com), [Noah Jung](https://github.com/giallo41), [Jujin Lee](https://github.com/zeze-zzz) and [Kyubyong Park](https://github.com/Kyubyong)
>* [Demo: Check out who your voice is like!](https://voice-vector-web.andabi.me/)

## Prologue
Everyone has their own voice.
The same voice will not exist from different people, but some people have similar voices while others do not.
This project aims to find individual voice vectors using [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb) dataset, which contains 1,251 Hollywood stars' 145,379 utterances.
The voice vectors are text-independent, meaning that any pair of utterances from same speaker has similar voice vectors.
Also the closer the vector distance is, the more voices are similar.

<p align="center"><img src="https://raw.githubusercontent.com/andabi/voice-vector/master/materials/title.png" width="60%"></p>

## Architectures
The architecture is based on a classification model.
The utterance inputted is classified as one of the Hollywood stars.
The objective function is simply a cross entropy between speaker labels from ground truth and predictions.
Eventually, the last layer's activation becomes the speaker's embedding.

The model architecture is structured as follows.
1. memory cell
    * CBHG module from [Tacotron](https://arxiv.org/abs/1703.10135) captures hidden features from sequential data.
2. embedding
    * memory cell's last output is projected by the size of embedding vector.
3. softmax
    * embedding is logits for each classes.

<p align="center"><img src="https://raw.githubusercontent.com/andabi/voice-vector/master/materials/arch.png" width="100%"></p>

## Training
* [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb) dataset used.
  * 1,251 Hollywood stars' 145,379 utterances
  * gender dist.: 690 males and 561 females
  * age dist.: 136, 351, 318, 210, and 236 for 20s, 30s, 40s, 50s, and over 60s respectively.
* text-independent
  * at each step, the speaker is arbitrarily selected.
  * for each speaker, the utterance inputted is randomly selected and cropped so that it does not matter to text.
* loss and train accuracy
  * <br><img src="https://raw.githubusercontent.com/andabi/voice-vector/master/materials/train_loss.png" width="40%"><img src="https://raw.githubusercontent.com/andabi/voice-vector/master/materials/train_acc.png" width="40%">

## Embedding
* [Common Voice](https://voice.mozilla.org/) dataset used for inference.
  * hundreds of thousands of English utterances from numerous voice contributors in the world.
* evaluation accuracy
  * <br><img src="https://raw.githubusercontent.com/andabi/voice-vector/master/materials/eval_acc.png" width="40%">
* embedding visualization using t-SNE
    * voices are well clustered by gender without any supervision in training.    
      * <p align="center"><img src="https://raw.githubusercontent.com/andabi/voice-vector/master/materials/embed_gender.png" width="100%"></p>
    * but we could not find any tendency toward age.
      * <p align="center"><img src="https://raw.githubusercontent.com/andabi/voice-vector/master/materials/embed_age.png" width="100%"></p>

## How to run?
### Requirements
  * python 2.7
  * tensorflow >= 1.1
  * numpy >= 1.11.1
  * librosa == 0.5.1
  * tensorpack == 0.8.0
### Settings
  * configurations are set in two YAML files.
  * `hparams/default.yaml` includes default settings for signal processing, model, training, evaluation and embedding.
  * `hparams/hparams.yaml` is for customizing the default settings in each experiment case.
### Runnable python files
  * `train.py` for training. 
    * run `python train.py some_case_name`
    * remote mode: utilizing more cores of remote server to load data and enqueue more quickly.
      * run `python train.py some_case_name -remote -port=1234` in local server.
      * run `python remote_dataflow.py some_case_name -dest_url=tcp://local-server-host:1234 -num_thread=12` in remote server.
  * `eval.py` for evaluation.
    * run `python eval.py some_case_name`
  * `embedding.py` for inference and getting embedding vectors.
    * run `python embedding.py some_case_name`
### Visualizations
  * Tensorboard
    * Scalars tab: loss, train accuracy, and eval accuracy.
    * Audio tab: sample audios of input speakers(wav) and predicted speakers(wav_pred)
    * Text tab: prediction texts with the following form: 'input-speaker-name (meta) -> predicted-speaker-name (meta)'
      * ex. sample-022653 (('female', 'fifties', 'england')) -> Max_Schneider (('M', '26', 'USA'))
  * t-SNE output file
    * outputs/embedding-[some_case_name].png

## Future works
* One-shot learning with triplet loss.

## References
* Nagrani, A., Chung, J. S., & Zisserman, A. (2017, June 27). [VoxCeleb: a large-scale speaker identification dataset](http://arxiv.org/abs/1706.08612v1). arXiv.org.
* Zhang, C., & Koishida, K. (2017). [End-to-End Text-Independent Speaker Verification with Triplet Loss on Short Utterances](http://www.isca-speech.org/archive/Interspeech_2017/abstracts/1608.html) (pp. 1487–1491). Presented at the Interspeech 2017, ISCA: ISCA. http://doi.org/10.21437/Interspeech.2017-1608
* Li, C., Ma, X., Jiang, B., Li, X., Zhang, X., Liu, X., et al. (2017, May 6). [Deep Speaker: an End-to-End Neural Speaker Embedding System](http://arxiv.org/abs/1705.02304v1). arXiv.org.
