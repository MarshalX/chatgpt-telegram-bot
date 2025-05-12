import hashlib
import logging
from typing import Dict, List

import dagger

from .plugin import Plugin

_IMAGE_NAME = 'python:3.12-slim'
# ref: https://wfhbrian.com/artificial-intelligence/mastering-chatgpts-code-interpreter-list-of-python-packages
_LIST_OF_PACKAGES = [
    'pandas',
    'numpy',
    'scipy',
    'xarray',
    'scikit-learn',
    'xgboost',
    'keras',
    'torch',
    'nltk',
    'spacy',
    'textblob',
    'gensim',
    'matplotlib',
    'seaborn',
    'plotly',
    'bokeh',
    'requests',
    'urllib3',
    'aiohttp',
    'beautifulsoup4',
    'keras',
    'torch',
    'pillow',
    'imageio',
    'opencv-python',
    'scikit-image',
    'librosa',
    'pyaudio',
    'soundfile',
    'openpyxl',
    'xlrd',
    'pyPDF2',
    'python-docx',
    'sqlalchemy',
    'psycopg2-binary',
    'mysql-connector-python',
    'flask',
    'django',
    'tornado',
    'quart',
    'pytest',
    'joblib',
    'pytz',
    'pyyaml',
]

# Create a cache key based on the packages list
_REQUIREMENTS_CACHE_KEY = hashlib.md5('\n'.join(_LIST_OF_PACKAGES).encode()).hexdigest()


class CodeExecutionPlugin(Plugin):
    """
    A plugin to execute Python code securely in an isolated container using Dagger.io
    """

    _bootstrap = False

    def __init__(self):
        super().__init__()

    def get_source_name(self) -> str:
        return 'Code'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'execute_python_code',
                    'description': 'Execute Python code securely in an isolated environment and return the results from stdout.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'code': {
                                'type': 'string',
                                'description': 'Python code to execute. The code must be complete and executable. The code must print the result to stdout.',
                            },
                            'timeout': {
                                'type': 'integer',
                                'description': 'Maximum execution time in seconds (default: 10; max: 120)',
                            },
                        },
                        'required': ['code', 'timeout'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        code = kwargs.get('code', '')
        if not code:
            return {'error': 'No code provided'}

        timeout = kwargs.get('timeout', 10)
        try:
            timeout = int(timeout)
        except ValueError:
            timeout = 10

        # Ensure timeout is reasonable
        timeout = min(max(1, timeout), 120)

        try:
            result = await self._run_code_with_dagger(code, timeout)
            return {'result': result}
        except Exception as e:
            logging.error(f'Error executing code: {str(e)}')
            return {'error': f'Execution error: {str(e)}'}

    @classmethod
    async def _run_code_with_dagger(cls, code: str, timeout: int) -> str:
        """Execute Python code in an isolated Dagger container"""

        # Connect to the Dagger Engine
        async with dagger.Connection(config=dagger.Config(execute_timeout=timeout)) as client:
            base_container = await cls._get_or_create_base_container(client)

            container = (
                base_container
                # Add the code to the container
                .with_new_file('/app/code_to_execute.py', contents=code)
                # Set the working directory
                .with_workdir('/app')
            )

            try:
                exec_result = await container.with_exec(['python', 'code_to_execute.py']).stdout()
                return exec_result

            except dagger.ExecError as e:
                # Handle execution errors
                logging.error(f'Execution error: {e.stderr or e.stdout or str(e)}')
                return f'Execution failed: {e.stderr or e.stdout or str(e)}'
            except Exception as e:
                # Handle any other errors
                return f'Error: {str(e)}'

    async def bootstrap(self):
        """Bootstrap the plugin by creating a base container with all packages installed"""
        async with dagger.Connection(config=dagger.Config()) as client:
            await self._get_or_create_base_container(client)
        CodeExecutionPlugin._bootstrap = True

    @classmethod
    async def _get_or_create_base_container(cls, client):
        """Get or create a cached container with all packages installed"""

        container = (
            client.container()
            .from_(_IMAGE_NAME)
            # .with_new_file(f'/app/requirements_{_REQUIREMENTS_CACHE_KEY}.txt', contents='\n'.join(_LIST_OF_PACKAGES))
            # .with_exec(['pip', 'install', '-r', f'/app/requirements_{_REQUIREMENTS_CACHE_KEY}.txt'])
            .with_workdir('/app')
        )

        return container
