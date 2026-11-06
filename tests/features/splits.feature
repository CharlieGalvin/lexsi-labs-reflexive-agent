Feature: Frozen task splits
  Every τ²-bench task belongs to exactly one split, and the splits never change.
  Test is used once, for the final results; nothing is ever trained on dev.

  Scenario: The splits have the agreed sizes
    Given the frozen splits
    Then there are 142 train, 36 dev and 100 test tasks
    And the dev split has 15 retail, 6 airline and 15 telecom tasks

  Scenario: No task is in two splits
    Given the frozen splits
    Then no task appears in more than one split, in any domain

  Scenario: A test task can never become training data
    Given a retail task from the test split
    When code checks whether the task may be trained on
    Then it is refused with an error saying it is a TEST task

  Scenario: A train task is allowed
    Given a retail task from the train split
    When code checks whether the task may be trained on
    Then it is allowed
