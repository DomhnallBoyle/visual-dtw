"""Flask Entry Point.

This script starts the Flask API
"""
import argparse

from main import create_app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8080)
    parser.add_argument('--drop_db', action='store_true')

    args = parser.parse_args()

    app = create_app(drop_db=args.drop_db)
    app.run(host='0.0.0.0', port=args.port, use_reloader=False)
