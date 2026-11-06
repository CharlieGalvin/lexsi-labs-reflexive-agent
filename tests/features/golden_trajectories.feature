Feature: Golden trajectories
  A teacher conversation that solved a train task becomes one training record for the student:
  the conversation as the agent saw it, in Aya's Action format, with the agent's turns as the
  learning targets. Every record is read back before it is kept.

  Background:
    Given a telecom conversation the teacher solved, in which the customer checked their phone

  Scenario: The student only sees what the agent saw
    When it is turned into a training record
    Then the customer's own phone actions are not in the record
    And the record starts with the system prompt, which lists the agent's tools

  Scenario: Tool calls and results use Aya's own format
    When it is turned into a training record
    Then each agent tool call is written as an "Action:" block
    And each tool result comes back as a customer-side turn inside <results> tags

  Scenario: Only the agent's turns are learning targets
    When it is turned into a training record
    Then every assistant turn is a learning target
    And no system, customer or tool-result turn is a learning target

  Scenario Outline: A damaged record is caught before it is kept
    When it is turned into a training record
    And the record's <part> is changed
    Then the check reports a problem mentioning "<problem>"

    Examples:
      | part                     | problem        |
      | tool call argument       | round trip     |
      | agent message            | round trip     |
      | tool result              | round trip     |
      | list of learning targets | train_on       |
      | system prompt            | system prompt  |

  Scenario: Only the first clean solution of a task is kept
    Given attempts of the same task that failed, retried a failed call, solved it cleanly, and solved it again
    When the golden trajectory is picked
    Then the third attempt is kept

  Scenario: A test task is never turned into a record
    When it is turned into a training record for the test split
    Then it is refused
