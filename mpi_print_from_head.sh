#!/bin/bash
if [[ $OMPI_COMM_WORLD_RANK == 0 ]]
then
  export COLUMNS=120
  "$@"
else
  "$@" 1>/dev/null 2>/dev/null
fi
