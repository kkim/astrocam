# AstroCam System Configuration

This file tracks local system-level configurations made for this project.

## Autostart Service (systemd)
The project is configured to start automatically on boot using a systemd service.

- **Service Name:** `astrocam.service`
- **Service File Path:** `/etc/systemd/system/astrocam.service` (copied from the project root)
- **Log Files:** `backend.log`, `frontend.log` in the project root.

### Management Commands

**To view status:**
```bash
sudo systemctl status astrocam.service
```

**To stop the service:**
```bash
sudo systemctl stop astrocam.service
```

**To disable autostart (but keep the service file):**
```bash
sudo systemctl disable astrocam.service
```

**To re-enable autostart:**
```bash
sudo systemctl enable astrocam.service
```

**To completely remove the service:**
```bash
sudo systemctl stop astrocam.service
sudo systemctl disable astrocam.service
sudo rm /etc/systemd/system/astrocam.service
sudo systemctl daemon-reload
```
