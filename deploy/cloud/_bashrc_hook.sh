# --- education AI: auto-resume on login (only if the app is not running) ---
if [ -f /root/education/education/deploy/cloud/resume.sh ]; then
  if ! curl -s -m2 -o /dev/null http://127.0.0.1:8001/health; then
    setsid bash /root/education/education/deploy/cloud/resume.sh > /root/education/education/deploy/cloud/logs/resume.log 2>&1 &
  fi
fi
