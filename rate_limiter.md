# Rate Limiter Documentation

## Overview

The `RateLimiter` class helps manage Telegram API rate limits to prevent 429 errors when sending messages or updating message content. This is especially important during streaming responses where many updates might occur in quick succession.

## Telegram API Rate Limits

Telegram imposes the following rate limits on bots:

- **Private chats**: Maximum of 1 message per second (with short bursts allowed)
- **Group chats**: Maximum of 20 messages per minute
- **Broadcast**: Maximum of about 30 messages per second overall

## Configuration Parameters

The rate limiter behavior can be customized through the following environment variables:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ENABLE_RATE_LIMIT` | Enable/disable rate limiting | `true` |
| `GROUP_RATE_LIMIT` | Maximum messages per minute in groups | `20` |
| `PRIVATE_RATE_LIMIT` | Minimum seconds between messages | `1.0` |
| `MAX_UPDATE_FREQUENCY` | Maximum streaming update frequency (seconds) | `0.5` |

## Key Methods

### `check_and_wait(chat_id, is_group)`

This method enforces the basic rate limits by:

1. Tracking the number of messages sent to a group chat per minute
2. Enforcing the minimum time between messages
3. Waiting when necessary to comply with limits
4. Returning `False` if a hard limit (group message limit) is reached

Usage:
```python
# Before sending any message
can_send = await rate_limiter.check_and_wait(chat_id, is_group_chat)
if can_send:
    # Send the message
else:
    # Skip or postpone sending
```

### `should_update(chat_id, is_group, current_length, prev_length, cutoff)`

This method determines whether to send streaming updates based on:

1. How significant the content change is (relative to a cutoff threshold)
2. How recently the last update was sent
3. Whether we're approaching group message rate limits
4. Chat type (more conservative in groups)

Usage:
```python
# During streaming, before each update
should_update = rate_limiter.should_update(
    chat_id, 
    is_group, 
    len(current_content), 
    len(previous_content), 
    cutoff_value
)
if should_update:
    # Proceed with sending the update
else:
    # Skip this intermediate update
```

## How It Works

### Rate Limiting Strategy

The rate limiter employs several strategies to prevent hitting Telegram's limits:

1. **Enforced Delays**: Ensures a minimum time between messages
2. **Group Message Tracking**: Counts messages per minute in group chats
3. **Smart Update Skipping**: Avoids unnecessary intermediate updates during streaming
4. **Adaptive Thresholds**: Becomes more selective when approaching rate limits
5. **Context-Aware**: More conservative in groups, more responsive in private chats

### Update Frequency Control

For streaming responses, the rate limiter intelligently decides when to skip updates:

- Updates are skipped if they occur too frequently
- Small content changes are ignored if the last update was recent
- Larger changes are prioritized (using the cutoff multiplier)
- Final updates (when streaming ends) are always sent

### Data Tracking

The rate limiter maintains several trackers to enforce limits:

- `last_update_time`: Last message time per chat
- `group_message_count`: Messages sent to each group in the current minute
- `group_minute_start`: When the current minute began for each group
- `last_stream_update`: Last streaming update time per chat

## Best Practices

1. Always check `can_send` before sending any message
2. For streaming updates, check `should_update` first, then `check_and_wait`
3. Always prioritize final messages (when streaming ends)
4. Be more conservative with updates in group chats
5. Provide meaningful progress updates while minimizing API calls

## Example Flow for Streaming Updates

```
START STREAMING
↓
Get initial content chunk
↓
Send first message (use check_and_wait)
↓
For each content update:
  ↓
  Check if update is worthwhile (use should_update)
  ↓
  If yes:
    Check if we can send now (use check_and_wait)
    ↓
    If yes:
      Send the update
    Else:
      Skip this update
  Else:
    Skip this update
↓
Send final message (use check_and_wait)
END STREAMING
```

This approach balances showing meaningful progress to the user while respecting Telegram's API limits. 