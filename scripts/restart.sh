nohup bash /root/MeshForge/scripts/deploy.sh > /root/deploy_meshforge.log 2>&1 &
echo RESTARTED
pgrep -fa deploy.sh
