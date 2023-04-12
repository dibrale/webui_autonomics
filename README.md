# Emotional State Mirroring via Parameter Adjustment

## Introduction

Interaction with an AI chat agent proceeds more smoothly when it can mirror the emotional state of the user. Adjusting text generation parameters can influence the formality, verbosity, excitement level and affect conveyed by the response. This extension for [oobabooga's text-generation-webui](https://github.com/oobabooga/text-generation-webui) locally queries [Emotion English DistilRoBERTa-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) in order to obtain a sentiment evaluation of the input, then calculates a score that is used to adjust model parameters within a given range.

Rather than conveying input sentiment directly via the prompt, this scoring system and subsequent parameter adjustment simulates an autonomic response to the input. This adjustment then helps the chat agent mirror user affect as it generates a response. Insofar as the literature allows, scoring function component calculations are inspired by known autonomic response curves to human emotion. The balance of the scoring function is a work in progress.

The parameter ranges built into the model are calibrated for [LLaMA-13B](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) variant models, but can be adjusted by the user at runtime.

## Installation

1. Make a new directory under text-generation-webui/extensions (eg.: text-generation-webui/extensions/webui-autonomics).
2. Copy *script.py* from this repository into the new directory.
3. Start `server.py` with the `--extensions webui-autonomics` flag.
4. The extension will download *DistilRoBERTa* to run it locally, and will then be ready for use.

## Usage

The UI of this extension looks like this:

![autonomic](https://user-images.githubusercontent.com/108030031/231602382-b77ce422-6703-4d15-b6d5-7d206d5154a6.png)


- Sliders allow the user to tune parameter ranges within which the model operates. Parameters that can be presently adjusted: `temperature`, `typical_p`, `repetition_penalty`, `encoder_repetition_penalty` and `penalty_alpha`
- Press the 'Autonomic Update' button prior to each output generation. The extension will then analyze the sentiment of the input textbox, aggregate the scores and calculate a new set of parameters. These are written to one of two buffer presets in an alternating fashion. The extension then loads the preset it wrote to so the new parameters will be used in the next generation call from the UI.
- The 'Print debug information to console' checkbox will print the sentiment scores, processed contributions and the final 'Autonomic coefficient' to console. The autonomic coefficient ranges from 0 (least stimulating input) to 1 (most stimulating) input.
