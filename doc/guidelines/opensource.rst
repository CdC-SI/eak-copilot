#########################
Open-Source collaboration
#########################

What is GitHub?
    Github is a platform for version control and collaboration.

What is Open Source?
    An open source software has public source code that anyone can inspect, modify, and enhance.

Workflow
========
#. :ref:`Pick up an issue <pick-issue>`
#. :ref:`Create a branch <create-branch>`
#. Write changes related to the issue and push it to the branch
#. :ref:`Commit <commit>`
#. :ref:`Do a pull request <pull-request>`
#. :ref:`Code review and feedback <code-review>`
#. :ref:`Merge branch <merge-branch>`
#. :ref:`Close issues <close-issues>`

Tutorial
========

.. _create-issues:

Create issues
-------------

.. admonition:: What are issues?
    :class: hint

    Issues are a way to track enhancements, tasks, or bugs for work on Github.

#. Identify a need or a problem in the project or EPIC.
#. Go to the project’s `Issues tab <https://github.com/CdC-SI/eak-copilot/issues>`_ and click "New issue", or go to the `project board <https://github.com/orgs/CdC-SI/projects/2>`_ and create a new issue in the backlog or the current sprint’s backlog.
#. Provide a clear, concise title and a detailed :ref:`description <issue-template>`.
#. If you are going to work on this issue, assign yourself to the issue immediately, else place the issue in the backlog (current sprint or general backlog) or in the current sprint’s "Ready" tab.
#. Additionally, add a label, priority, size, sprint iteration and milestone.

.. admonition:: Best practices
    :class: important

    * Label issues appropriately (e.g., bug, feature, etc.).
    * Assign issues for specific tracking and responsibility.

|

.. _create-epics:

Create EPICs and manage issues
------------------------------

.. admonition:: What is an EPIC?
    :class: hint

    An EPIC is a larger body of work that can be broken down into smaller tasks (issues), typically used to group related features or improvements.

Creating an EPIC
    #. Use a GitHub issue to represent the EPIC (name it **"EPIC: name_of_epic"**).
    #. Provide a detailed description that outlines the objectives, potential tasks, and expected outcomes.

Incorporating issues into an EPIC
    #. Create individual issues for tasks that fall under the EPIC’s scope.
    #. Link issues to the EPIC by mentioning the task's issue number (eg. #123) in each EPIC description.

.. admonition:: Best practices
    :class: important

    * Clearly define the scope and goals of the EPIC.
    * Regularly review and update the EPIC and its associated issues to reflect changes in scope, priorities, or deadlines.
    * Use labels and milestones to organize and track progress on the EPIC and related issues.

|

.. _pick-issue:

Pick up an issue
----------------

#. Browse the current sprint's "Ready" tab for unassigned issues.
#. Assign yourself to the issue.
#. Place the issue in the "In progress" tab.

.. admonition:: Best practices
    :class: important

    * Choose issues that match your skills.
    * Communicate openly with project maintainers about your plans.

|

.. _create-branch:

Create a branch
---------------

.. admonition:: What is a branch?
    :class: hint

    A branch is a version of the repository that diverges from the main working project.

Use Github or Git command line

.. code-block:: console

    git checkout -b branch_name

.. admonition:: Best practices
    :class: important

    * Name branches clearly (e.g., feature/#123-add-query-autocomplete, bugfix/#147-chatbar-display).
    * Each issue should have its own branch to keep changes organized.
    * Keep your branch regularly updated with main: ``git pull`` (does a merge).

|

.. _commit:

Commit (and link to issues)
---------------------------

.. admonition:: What is a commit?
    :class: hint

    A commit records changes to one or more files in your branch.

#. Make changes in your branch
#. Run ``git add``
#. Run ``git commit –m "your_commit_ message"`` with a descriptive message.

Linking Commits to Issues
    Use keywords in your commit message to link the commit to an issue (e.g., "fixes #123", "closes #124").

    This automatically closes the issue when the commit is merged into the default branch (after review).

.. admonition:: Best practices
    :class: important

    * Use clear, descriptive commit messages.
    * Commit often to document your progress and changes.

.. seealso::

    `Using keywords in issues and pull requests - GitHub Docs <https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests>`_

|

.. _pull-request:

Pull Request
------------

.. admonition:: What is a Pull Request (PR)?
    :class: hint

    A way to propose changes from your branch to the main project.

#. Push your branch to GitHub.
#. Open a pull request via the GitHub interface.
#. Provide a :ref:`context <pr-template>` and link the issue you are addressing.

.. admonition:: Best practices
    :class: important

    * Review your changes before submitting a PR.
    * Request review from maintainers or other contributors.
    * Ensure at least one acceptance of review before merge and close issue.
    * Use ``fixes`` for bugfix, ``closes`` for feature.

.. seealso::

    * `Creating a pull request - GitHub Doc <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_
    * `Linking a pull request to an issue - GitHub Docs <https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue>`_

|

.. _code-review:

Code review and feedback
------------------------

Other contributors review your changes, suggest improvements, or approve the changes.

* Be open to feedback and ready to make further changes.
* Respond to comments to explain decisions if necessary.
* Anyone can review code if they feel comfortable with it.

.. admonition:: Best practices
    :class: important

    * Be respectful and constructive in comments.
    * Comment in diff view in Github interface.

|

.. _merge-branch:

Merge branch
------------

.. admonition:: What is merging?
    :class: hint

        Merging is incorporating the changes from one branch into another, typically into the main branch.

* If your PR is approved, a project maintainer will merge the branch.
* GitHub often allows automatic merging if there are no conflicts.

.. admonition:: Best practices
    :class: important

    * Keep your branch up to date with the main branch to minimize conflicts.

|

.. _close-issues:

Close issues
------------

Issues should be closed after the related changes are merged and the problem has been solved.

* Use the GitHub interface to close the issue, often automatically linked by mentioning in the PR (e.g., "fixes #123").

.. admonition:: Best practices
    :class: important

    * Confirm the issue is fully resolved before closing.
    * Provide a closing comment to explain the resolution.

|

Templates
=========


.. _issue-template:

Issue
-----

Please use the following template to :ref:`submit an issue <create-issues>`.

.. code::

    **Issue Title**

    Setup /get_docs endpoint in backend with FastAPI.

    **Description**

    Setup a document retrieval endpoint in backend.

    Do a semantic similarity match (cosine similarity) on indexed documents in postgres vector DB.

    Takes as input a user query such as: {"query": query} and returns a response such as {"contextDocs": docs, "sourceUrl": url}.

    Will be called from the frontend when a user inputs a query that doesn't match an autocomplete suggestion.

    **Steps to Reproduce**

    For a feature, skip this step. For a bug, detail the steps to reproduce the bug.

        Go to '...'
        Click on '....'
        Scroll down to '....'
        See error

    **Expected Behavior**

    For a feature, skip this step. For a bug, describe what you expected to happen when following the steps above.

    **Actual Behavior**

    For a feature, skip this step. For a bug, describe what actually happened. Include screenshots or animated GIFs if applicable.

    **Possible Solution**

    For a feature, skip this step. For a bug, include any suggestions on a fix or a reason for the bug.

    **Context**

    For a feature, skip this step. For a bug, provide any context or additional information that might be helpful. This can include the environment in which the issue occurred (device, OS, browser, specific software versions).

    **Your Environment**

    For a feature, skip this step. For a bug, specify any relevant details about your setup:

        Version used:
        Operating System and version:
        Browser and version:
        Other relevant software or hardware:

    **Logs and Additional Information**

    For a feature, skip this step. For a bug, include any error logs or any other information that might be relevant. You can format error logs or code snippets using Markdown code blocks.


.. _pr-template:

Pull Request
------------

Please use the following template to :ref:`create a Pull Request <pull-request>`.

.. code::

    **Overview**

    Added a simple RAG functionality to the EAK-Copilot.

    **Issue Reference:**

        - EPIC: setup baseline RAG #79
        - install postgres vectorDB extension #105
        - index baseline vector data in vectorDB #107
        - setup base retriever #97
        - setup /get_docs endpoint in backend #134
        - integrate rag to frontend #110
        - display RAG source URL in chat conversation #135
        - update README #112

    **Testing**

        - Tested endpoints manually through swagger docs at localhost:8010/docs.
        - Removed the db/data folder to index docs from scratch.
        - Ran rag/app/test_semantic_search.py.

    **Screenshots**

    ![Eg. A GUI change](/screenshot.png)

    **Additional Notes**

        - Crawling/Scraping/Indexing of fedlex.ch, ahv-iv.ch, zas.admin.ch and bsv.admin.ch will be implemented in a future sprint. For the moment only dummy data is indexed.
        - Auto data crawling/scraping/indexing will be configured from the main config.yaml file in the future (for the moment this process is manual).