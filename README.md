# b25 - Understanding song relations

This repository contains the codebase for exploring how word embedding techniques can be adapted to model music recommendation systems. By drawing inspiration from natural language processing, the project treats playlists as collections of co-occurring songs, enabling the exploration of novel embedding approaches tailored to non-sequential data.

The project provides implementations of several established embedding models alongside innovative adaptations designed specifically for recommendation contexts. It serves as both a research framework and a practical tool for investigating how semantic relationships among songs can be captured and leveraged for improved recommendation performance.

## to do's
- getting all the f1 scores for every model with the v3 data
- putting together the paper for this repo
- refactoring CBOE 
- write one pretty package out of the entity embedding trainings
    - inheritance and everything you know

## notes 
training got faster with SGE but still really really slow... 
SGE-64-20 took about 1.5 Days... 
SGE-256-20 taking about 2.5 Days even after implementing multi core training

Need to convert the already trained models into the new models format with the easy loading and saving instead of the wierd way chat gpt did dirty...