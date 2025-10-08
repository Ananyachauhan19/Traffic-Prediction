
import os

if __name__ == '__main__':
	# Run uvicorn server
	import uvicorn

	host = os.environ.get('HOST', '0.0.0.0')
	port = int(os.environ.get('PORT', 8000))
	uvicorn.run('backend.app.api:app', host=host, port=port, reload=True)
