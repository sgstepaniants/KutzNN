#!/bin/sh

awk -F, '{print $1}' < $1 > $2
