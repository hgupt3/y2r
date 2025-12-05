#!/usr/bin/env python3
"""
Simple HTTP server to serve the visualization viewer.

Usage:
    python serve_viewer.py [port]
    
Automatically detects 2D/3D mode from tracker_params.yaml config.
"""

import http.server
import socketserver
import os
import sys
import socket
import yaml
from pathlib import Path


class ReusableTCPServer(socketserver.TCPServer):
    """TCP server that allows address reuse."""
    allow_reuse_address = True


def get_local_ip():
    """Get local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"


def get_viewer_mode():
    """Read visualization mode from config file."""
    config_path = Path(__file__).parent / 'config' / 'tracker_params.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('visualization', {}).get('mode', '2d')
    except:
        return '2d'


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    
    # Detect mode from config
    mode = get_viewer_mode()
    viewer_file = 'viewer_3d.html' if mode == '3d' else 'viewer_2d.html'
    
    # Change to viewer directory
    viewer_dir = Path(__file__).parent / 'viewer'
    os.chdir(viewer_dir)
    
    # Write config.js for index.html to read
    with open('config.js', 'w') as f:
        f.write(f'window.VIEWER_MODE = "{mode}";')
    
    handler = http.server.SimpleHTTPRequestHandler
    local_ip = get_local_ip()
    
    with ReusableTCPServer(("", port), handler) as httpd:
        print(f"Mode: {mode.upper()} (from config)")
        print(f"Viewer: http://{local_ip}:{port}/{viewer_file}")
        print(f"Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == '__main__':
    main()

