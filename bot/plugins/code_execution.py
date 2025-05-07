import io
import logging
from typing import Dict, List

import dagger

from .plugin import Plugin


class CodeExecutionPlugin(Plugin):
    """
    A plugin to execute Python code securely in an isolated container using Dagger.io
    """

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
                                'description': 'Python code to execute. The code should be complete and executable. Prints to stdout will be returned.',
                            },
                            'timeout': {
                                'type': 'integer',
                                'description': 'Maximum execution time in seconds (default: 10)',
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
        timeout = min(max(1, timeout), 30)

        try:
            result = await self._run_code_with_dagger(code, timeout)
            return {'result': result}
        except Exception as e:
            logging.error(f'Error executing code: {str(e)}')
            return {'error': f'Execution error: {str(e)}'}

    @staticmethod
    async def _run_code_with_dagger(code: str, timeout: int) -> str:
        """Execute Python code in an isolated Dagger container"""

        # Connect to the Dagger Engine
        async with dagger.Connection(config=dagger.Config(execute_timeout=timeout, log_output=io.StringIO())) as client:
            # Get python container
            container = (
                client.container()
                .from_('python:3.13-slim')
                # Add the code to the container
                .with_new_file('/app/code_to_execute.py', contents=code)
                # Set the working directory
                .with_workdir('/app')
            )

            try:
                # Execute the Python script with stdout and stderr capture
                exec_result = await container.with_exec(['python', 'code_to_execute.py']).stdout()

                # Return the combined output
                return exec_result

            except dagger.ExecError as e:
                # Handle execution errors
                return f'Execution failed: {e.stderr or e.stdout or str(e)}'
            except Exception as e:
                # Handle any other errors
                return f'Error: {str(e)}'
