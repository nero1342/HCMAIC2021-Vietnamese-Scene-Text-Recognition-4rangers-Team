#!/usr/bin/env bash

# gather models
cat ../configs/kie/*/*.md | sed '$a\\n' | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Key Information Extraction Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >kie_models.md
cat ../configs/textdet/*/*.md | sed '$a\\n' | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Text Detection Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >textdet_models.md
cat ../configs/textrecog/*/*.md | sed '$a\\n' | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Text Recognition Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >textrecog_models.md
cat ../configs/ner/*/*.md | sed '$a\\n' | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Named Entity Recognition Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >ner_models.md

# replace special symbols in demo.md
cp ../demo/README.md demo.md
sed -i 's/:heavy_check_mark:/Yes/g' demo.md && sed -i 's/:x:/No/g' demo.md
