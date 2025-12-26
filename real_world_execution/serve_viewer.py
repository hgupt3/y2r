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


def get_viewer_config():
    """Read visualization config from config file."""
    config_path = Path(__file__).parent / 'config' / 'tracker_params.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        mode = config.get('visualization', {}).get('mode', '2d')
        text_prompt = config.get('perception', {}).get('text_prompt', None)
        return mode, text_prompt
    except:
        return '2d', None


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    
    # Detect mode and prompt from config
    mode, text_prompt = get_viewer_config()
    viewer_file = 'viewer_3d.html' if mode == '3d' else 'viewer_2d.html'
    
    # Change to viewer directory
    viewer_dir = Path(__file__).parent / 'viewer'
    os.chdir(viewer_dir)
    
    # Write config.js for viewers to read
    prompt_js = f'"{text_prompt}"' if text_prompt else 'null'
    with open('config.js', 'w') as f:
        f.write(f'window.VIEWER_MODE = "{mode}";\n')
        f.write(f'window.DEFAULT_PROMPT = {prompt_js};\n')
    
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

