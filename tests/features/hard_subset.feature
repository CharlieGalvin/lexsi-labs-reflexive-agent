Feature: The hard subset
  `just hard-subset` sorts every train task by how the untuned student (Arm A) did and whether
  the teacher's verified fix exists. Healing trains on the healable tasks; the others show what
  is beyond the teacher and what must keep working. Dev and test tasks are never in it.

  Scenario Outline: Each train task lands in one group
    Given a retail train task that Arm A <result>
    And the teacher <fix>
    When the hard subset is built
    Then the task is in the "<group>" group

    Examples:
      | result | fix                | group            |
      | failed | solved it          | healable         |
      | failed | never solved it    | teacher_unsolved |
      | passed | solved it          | solved_by_aya    |
      | passed | never solved it    | solved_by_aya    |

  Scenario: Calling the customer's own phone tool is a Tool failure
    Given a telecom train task in which Arm A called one of the customer's phone tools
    When its failure labels are read
    Then they are "TOOL" and marked as a customer tool

  Scenario: Dev tasks are left out
    Given Arm A's records include a retail dev task
    When the hard subset is built
    Then the dev task isn't in it and is counted as left out

  Scenario: A test task stops the build
    Given Arm A's records include a retail test task
    When the hard subset is built
    Then the build stops, naming the test task
