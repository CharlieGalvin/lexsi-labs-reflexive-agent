Feature: Publishing the golden trajectories
  `just golden-publish` uploads data/golden/v1 to the team's Hugging Face organisation. It
  checks everything first and uploads nothing if any check fails. The dataset stays private
  until the team decides to release it, and a published version is never silently replaced.

  Background:
    Given a golden trajectories folder that passed its build checks

  Scenario: A first upload creates a private dataset and tags it
    Given the dataset doesn't exist yet on Hugging Face
    When it is published
    Then the upload succeeds
    And the dataset is created private
    And the data card, both record files and the splits are uploaded
    And the upload is tagged "v1"

  Scenario: It refuses to upload to a public dataset
    Given the dataset already exists on Hugging Face and is public
    When it is published
    Then nothing is uploaded

  Scenario: It refuses to replace a published version
    Given the dataset already exists on Hugging Face, privately, with the tag "v1"
    When it is published
    Then nothing is uploaded

  Scenario: It refuses files that changed since they were built and checked
    Given the dataset doesn't exist yet on Hugging Face
    But a record file was edited after the build
    When it is published
    Then nothing is uploaded

  Scenario: A record from the test split is never published
    Given one of the training records is from the test split
    When the records are checked
    Then the check reports a problem mentioning "test"

  Scenario: A new version adds the hard subset and leaves v1 alone
    Given the dataset already exists on Hugging Face, privately, with the tag "v1"
    And a hard subset built from these golden trajectories
    When it is published as "v1.1"
    Then the upload succeeds
    And the hard subset is uploaded with the rest, in one commit
    And the upload is tagged "v1.1"
    And the tag "v1" is left where it was
