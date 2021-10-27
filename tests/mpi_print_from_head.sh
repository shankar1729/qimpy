#!/bin/bash
if [[ $OMPI_COMM_WORLD_RANK == 0 ]]
then
  "$@"
else
  "$@" 1>/dev/null 2>/dev/null
fi
