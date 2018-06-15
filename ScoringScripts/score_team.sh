#!/bin/sh

team=$(basename $1)

for f in a b c d
do
    # Burn-in scores
    if [ -e "$team/${f}_burn_in.csv" ]; then
        res=$(python scripts/score.py $team/${f}_burn_in.csv test/$f.csv)
        echo "$team,$f,yes,$res"
    fi

    # results scores
    if [ -e "$team/${f}_no_burn_in.csv" ]; then
       res=$(python scripts/score.py $team/${f}_no_burn_in.csv test/$f.csv)
       echo "$team,$f,no,$res"
    fi
done
