from collections import defaultdict
from typing import Any, Dict, List

from .plugin import Plugin


class SequentialThinkingPlugin(Plugin):
    """
    A plugin for dynamic and reflective problem-solving through sequential thoughts
    """

    def __init__(self):
        self.thought_history_counter = defaultdict(int)
        self.branches_counter = defaultdict(lambda: defaultdict(int))

    def get_source_name(self) -> str:
        return 'SequentialThinking'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'sequential_thinking',
                    'description': """A detailed tool for dynamic and reflective problem-solving through thoughts.
This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
Each thought can build on, question, or revise previous insights as understanding deepens.

When to use this tool:
- Breaking down complex problems into steps
- Planning and design with room for revision
- Analysis that might need course correction
- Problems where the full scope might not be clear initially
- Problems that require a multi-step solution
- Tasks that need to maintain context over multiple steps
- Situations where irrelevant information needs to be filtered out

Key features:
- You can adjust total_thoughts up or down as you progress
- You can question or revise previous thoughts
- You can add more thoughts even after reaching what seemed like the end
- You can express uncertainty and explore alternative approaches
- Not every thought needs to build linearly - you can branch or backtrack
- Generates a solution hypothesis
- Verifies the hypothesis based on the Chain of Thought steps
- Repeats the process until satisfied
- Provides a correct answer

Parameters explained:
- thought: Your current thinking step, which can include:
* Regular analytical steps
* Revisions of previous thoughts
* Questions about previous decisions
* Realizations about needing more analysis
* Changes in approach
* Hypothesis generation
* Hypothesis verification
- next_thought_needed: True if you need more thinking, even if at what seemed like the end
- thought_number: Current number in sequence (can go beyond initial total if needed)
- total_thoughts: Current estimate of thoughts needed (can be adjusted up/down)
- is_revision: A boolean indicating if this thought revises previous thinking
- revises_thought: If is_revision is true, which thought number is being reconsidered
- branch_from_thought: If branching, which thought number is the branching point
- branch_id: Identifier for the current branch (if any)
- needs_more_thoughts: If reaching end but realizing more thoughts needed

You should:
1. Start with an initial estimate of needed thoughts, but be ready to adjust
2. Feel free to question or revise previous thoughts
3. Don't hesitate to add more thoughts if needed, even at the "end"
4. Express uncertainty when present
5. Mark thoughts that revise previous thinking or branch into new paths
6. Ignore information that is irrelevant to the current step
7. Generate a solution hypothesis when appropriate
8. Verify the hypothesis based on the Chain of Thought steps
9. Repeat the process until satisfied with the solution
10. Provide a single, ideally correct answer as the final output
11. Only set next_thought_needed to false when truly done and a satisfactory answer is reached""",
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'thought': {'type': 'string', 'description': 'Your current thinking step'},
                            'thought_number': {
                                'type': 'integer',
                                'description': 'Current thought number (minimum 1)',
                            },
                            'total_thoughts': {
                                'type': 'integer',
                                'description': 'Estimated total thoughts needed (minimum 1)',
                            },
                            'next_thought_needed': {
                                'type': 'boolean',
                                'description': 'Whether another thought step is needed',
                            },
                            'is_revision': {'type': 'boolean', 'description': 'Whether this revises previous thinking'},
                            'revises_thought': {
                                'type': 'integer',
                                'description': 'Which thought is being reconsidered (minimum 1)',
                            },
                            'branch_from_thought': {
                                'type': 'integer',
                                'description': 'Branching point thought number (minimum 1)',
                            },
                            'branch_id': {'type': 'string', 'description': 'Branch identifier'},
                            'needs_more_thoughts': {'type': 'boolean', 'description': 'If more thoughts are needed'},
                        },
                        'required': [
                            'thought',
                            'thought_number',
                            'total_thoughts',
                            'next_thought_needed',
                            'is_revision',
                            'revises_thought',
                            'branch_from_thought',
                            'branch_id',
                            'needs_more_thoughts',
                        ],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ]

    @staticmethod
    def _validate_thought_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the input thought data"""
        if not data.get('chat_id'):
            raise ValueError('chat_id is required')

        if not data.get('thought') or not isinstance(data['thought'], str):
            raise ValueError('Invalid thought: must be a string')
        if not data.get('thought_number') or not isinstance(data['thought_number'], int):
            raise ValueError('Invalid thought_number: must be an integer')
        if not data.get('total_thoughts') or not isinstance(data['total_thoughts'], int):
            raise ValueError('Invalid total_thoughts: must be an integer')
        if not isinstance(data.get('next_thought_needed'), bool):
            raise ValueError('Invalid next_thought_needed: must be a boolean')

        return {
            'chat_id': data['chat_id'],
            'thought': data['thought'],
            'thought_number': data['thought_number'],
            'total_thoughts': data['total_thoughts'],
            'next_thought_needed': data['next_thought_needed'],
            'is_revision': data.get('is_revision'),
            'revises_thought': data.get('revises_thought'),
            'branch_from_thought': data.get('branch_from_thought'),
            'branch_id': data.get('branch_id'),
            'needs_more_thoughts': data.get('needs_more_thoughts'),
        }

    @staticmethod
    def _format_thought(thought_data: Dict[str, Any]) -> str:
        """Format a thought for display"""
        thought_number = thought_data['thought_number']
        total_thoughts = thought_data['total_thoughts']
        thought = thought_data['thought']
        is_revision = thought_data.get('is_revision')
        revises_thought = thought_data.get('revises_thought')
        branch_from_thought = thought_data.get('branch_from_thought')
        branch_id = thought_data.get('branch_id')

        if is_revision:
            prefix = '🔄 Revision'
            context = f' (revising thought {revises_thought})'
        elif branch_from_thought:
            prefix = '🌿 Branch'
            context = f' (from thought {branch_from_thought}, ID: {branch_id})'
        else:
            prefix = '💭 Thought'
            context = ''

        header = f'{prefix} {thought_number}/{total_thoughts}{context}'

        return f"""
Headers: {header}
Thought: {thought}
"""

    async def execute(self, function_name: str, helper, **kwargs) -> Dict[str, Any]:
        """Execute the sequential thinking function"""
        try:
            validated_input = self._validate_thought_data(kwargs)
            chat_id = validated_input['chat_id']

            # Adjust total_thoughts if needed
            if validated_input['thought_number'] > validated_input['total_thoughts']:
                validated_input['total_thoughts'] = validated_input['thought_number']

            # Store the thought in history
            self.thought_history_counter[chat_id] += 1

            # Handle branching
            if validated_input.get('branch_from_thought') and validated_input.get('branch_id'):
                branch_id = validated_input['branch_id']
                self.branches_counter[chat_id][branch_id] += 1

            # Format thought for display and log it
            formatted_thought = self._format_thought(validated_input)
            print(formatted_thought)  # Log the thought to console

            # Return the result
            return {
                'result': {
                    'thought_number': validated_input['thought_number'],
                    'total_thoughts': validated_input['total_thoughts'],
                    'next_thought_needed': validated_input['next_thought_needed'],
                    'branches': list(self.branches_counter[chat_id].keys()),
                    'thought_history_length': self.thought_history_counter[chat_id],
                }
            }
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}
