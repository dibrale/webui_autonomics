# Emotional State Mirroring via Parameter Adjustment

## Introduction

Interaction with an AI chat agent proceeds more smoothly when it can mirror the emotional state of the user. Adjusting text generation parameters can influence the formality, verbosity, excitement level and affect conveyed by the response. This extension for [oobabooga's text-generation-webui](https://github.com/oobabooga/text-generation-webui) locally queries [Emotion English DistilRoBERTa-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) in order to obtain a sentiment evaluation of the input, then calculates a score that is used to adjust model parameters within a given range.

Rather than conveying input sentiment directly via the prompt, this scoring system and subsequent parameter adjustment simulates an autonomic response to the input. This adjustment then helps the chat agent mirror user affect as it generates a response. Insofar as the literature allows, scoring function component calculations are inspired by known autonomic response curves to human emotion. The balance of the scoring function is a work in progress.

The parameter ranges built into the model are calibrated for [LLaMA-13B](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) variant models, but can be adjusted by the user at runtime.

## Installation

1. Make a new directory under *text-generation-webui/extensions* (eg.: *text-generation-webui/extensions/autonomics*).
2. Copy *script.py* from this repository into the new directory.
3. Start `server.py` with the `--extensions autonomics` flag.
4. The extension will download *DistilRoBERTa* to run it locally, and will then be ready for use. It will proceed to create the *text-generation-webui/parameter_ranges/* directory if it is not present, as well as the files *Autonomic_Buffer_A.txt* and *Autonomic_Buffer_B.txt* in the text-generation-webui/presets/ directory when it performs parameter updates.  

## Usage

The UI of this extension, shown here with an abbreviated number of parameter sliders, looks like this:

![autonomic2](https://user-images.githubusercontent.com/108030031/232086809-57398b01-1412-4955-81f0-4adf21ba48c4.png)

- Press the 'Autonomic Update' button prior to each output generation. The extension will then analyze the sentiment of the input textbox contents, aggregate the scores and calculate a new set of parameters. These are written to one of two buffer presets in an alternating fashion. The extension then loads the preset it wrote to so the new parameters will be used in the next generation call from the UI.
- The 'Print debug information to console' checkbox will print the sentiment scores, processed contributions and the final 'Autonomic coefficient' to console. The autonomic coefficient ranges from 0 (least stimulating input) to 1 (most stimulating) input.
- Sliders allow the user to tune parameter ranges within which the model operates. Numerical parameters that can be presently adjusted: `temperature`, `top_p`, `typical_p`, `top_k`, `epsilon_cutoff`, `eta_cutoff`, `repetition_penalty`, `repetition_penalty_range`, `encoder_repetition_penalty`, `penalty_alpha`, `no_repeat_ngram_size`, `min_length`, `tfs`, `top_a`, `max_new_tokens`, `generation_attempts`, `num_beams`, `length_penalty`, `mirostat_mode`, `mirostat_tau`, `mirostat_eta`
- The parameters `early_stopping` and `do_sample` appear as checkboxes. These are not dynamically adjusted.
- Parameter ranges along with the 'Print debug information to console' state can be loaded from a dropdown menu below the 'Parameter Ranges' slider group. These are read from the *text-generation-webui/parameter_ranges/* directory.
- Parameter ranges can be saved by entering a name for the range in the 'Parameter range name' textbox, then clicking the 'Save' button. A *.yaml file with the new name as a root will be created in the *text-generation-webui/parameter_ranges/* directory. The state of the 'Print debug information to console' toggle is also stored in this file.
