Feature: Scoring a conversation
  A conversation passes only if τ²-bench's own evaluator passes it, checking the final state
  of the databases rather than the agent's words. A run that broke (an API failure, a crash)
  says nothing about the model: it is reported apart, never counted as the model failing.

  Scenario Outline: The task's correct actions pass
    Given the correct actions for <domain> task <task>
    When they are scored by τ²-bench's evaluator
    Then the conversation passes with a reward of 1

    Examples:
      | domain  | task |
      | retail  | 0    |
      | airline | 1    |

  Scenario: Leaving out the last change to the database fails
    Given the correct actions for the first retail task that changes the database
    But without the last change to the database
    When they are scored by τ²-bench's evaluator
    Then the conversation fails with a reward of 0

  Scenario Outline: A conversation cut short fails, even with the correct actions
    Given the correct actions for the first telecom task
    When they are scored as a conversation that ended with "<ending>"
    Then the conversation fails with a reward of 0
    And the reason says it ended prematurely

    Examples:
      | ending                  |
      | max_steps               |
      | too_many_errors         |
      | context_window_exceeded |

  Scenario Outline: A broken run is reported apart, not as a model failure
    Given a saved conversation that <what happened>
    Then it counts as a broken run
    And it is never picked as a golden trajectory

    Examples:
      | what happened                          |
      | crashed with an unexpected error       |
      | stopped because of an API error        |
      | hit an infrastructure error            |

  Scenario: A conversation that ended normally is a real result
    Given a saved conversation that ended normally, with the customer saying goodbye
    Then it does not count as a broken run
