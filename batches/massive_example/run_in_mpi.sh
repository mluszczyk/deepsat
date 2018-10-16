#!/usr/bin/env bash                                                                                                                                                                                               set -e

module load plgrid/tools/python/3.6.5
module unload plgrid/libs/mesa
module load tools/pro-viz

.  /net/software/local/pro-viz/1.2/pro-viz-client/run_X

DISPLAYNUM=`echo $VGL_DISPLAY | cut -d: -f2`
echo "Xorg display number is: $VGL_DISPLAY"
VNCDISPLAY=$((${DISPLAYNUM}+10))
export DISPLAY=:$VNCDISPLAY
echo "DISPLAY=$DISPLAY"

echo "starting TurboVNC on $DISPLAY"
LOCK=/tmp/.X${VNCDISPLAY}-lock
SOCK=/tmp/.X11-unix/X${VNCDISPLAY}
[ -f $LOCK ] && rm -v ${LOCK}
[ -S $SOCK ] && rm -v ${SOCK}
/opt/TurboVNC/bin/vncserver $DISPLAY &

export CARLA_ROOT=/net/archive/groups/plggluna/kg/carla-wolf/
export PYTHONPATH=$CARLA_ROOT/PythonClient:$PYTHONPATH
export LD_LIBRARY_PATH=/net/software/local/libm/2.27/lib:$LD_LIBRARY_PATH

export SHA=$(git rev-parse HEAD || echo "unknown")

if [ "$OMPI_COMM_WORLD_RANK" = "0" ]
then
    export NEPTUNE_TOKEN_PATH=$PWD
    pipenv run neptune run --open-webbrowser false --config $NEPTUNE_CONFIG_FILE $PYTHON_MAIN $EXP_KWARGS
else
    pipenv run neptune run --open-webbrowser false --offline --config $NEPTUNE_CONFIG_FILE $PYTHON_MAIN $EXP_KWARGS
fi
