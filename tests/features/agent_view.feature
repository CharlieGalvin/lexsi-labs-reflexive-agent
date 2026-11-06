Feature: What the agent sees
  The agent sees its system prompt, the customer's messages and the results of its own tool
  calls. In telecom the customer also acts on their own phone; the agent never sees those
  actions or their results. Training records and the hallucination classifier both use this
  view, so neither learns from, nor judges by, something the agent couldn't see.

  Background:
    Given a telecom conversation in which the customer checked their own phone

  Scenario: The customer's phone actions are hidden from the agent
    When the agent's view is rebuilt from the conversation
    Then it has no customer tool calls and none of their results
    And it still has every message the customer wrote
    And it still has the agent's own tool calls and their results

  Scenario: The classifier judges the agent on the same view
    When the classifier rebuilds what the agent saw
    Then it has no customer tool calls and none of their results
    And it still has the agent's own tool calls and their results

  Scenario: The agent's view survives the end of the conversation
    The τ²-bench run clears what the agent saw when the conversation ends. Our wrapper keeps a
    copy, so the saved view is complete even when the agent itself ends the conversation.

    Given a retail conversation in which the agent made the task's correct calls, then ended it
    Then the conversation is over
    And the saved view still starts with the system prompt
    And it still has the result of every call the agent made
