Are User Stories and Stories the same thing?
============================================

We often abbreviate the reference to a User Story as simply a Story. So,
from this point of view there is no difference. They are the same thing.
However, the concept of a story has evolved over time. A user story is
technically an articulation of something a person or role wants from a
piece of software for some reason. Hence, we see the common format:

> As a \<role\>, I want to \<action\> so that \<some result\>

While technically not correct, the concept of a user story has evolved
into representing some small requirement that can be implemented within
an iteration. These often represent some piece of functionality, but not
always, and not always initiated by a user.

While many pieces of software need to focus on delivering "value" to
users, user stories don't always cover everything that needs to be done
to implement a piece of software. One mechanism to overcome this is to
reuse the concept to represent work that is not necessarily a user
invocation of the software.

This is particularly true if a tracking an implementation specific piece
of functionality that is white box in nature. Ideally this behavior
should trace to some "user story" which is always black box in nature,
but it is often not easy to do. For instance some common behavior that
applies to or will apply to multiple user stories.

Is a Story a Requirement?
=========================

The simple answer is yes. It identifies something (often many things)
that is required of a piece of software.

However, in reality it is not that simple for a number of reasons. A
user story is a cross between a requirement and a work item. If/when it
is important to track and manage requirements it is helpful to not
discard the requirement when the implementation is complete. There are
several strategies for doing this. It takes some evaluation of a
situation to determine which fits best.

Most stories are not simple declarative statements. Hence, they often
cover what would traditionally be considered multiple requirements.
However, it can be useful to simply look at a story as single usage of
the software. The value is in the implementation of the story not always
sub parts of a story unless they can be stories in and of themselves.

Acceptance Criteria
-------------------

Further, a story generally contains acceptance criteria. In this context
acceptance criteria are how the software will be evaluated to determine
if the story has been implemented. To this end, acceptance criteria
often contain the specific details for a story. Certainly, a coder
should code to the acceptance criteria. For the acceptance criteria
represent the correct interpretation of the story and what it means to
have completed the story. These acceptance criteria are essentially a
detailed set of requirements that represent the story.

User Stories are not Features
-----------------------------

While user stories represent the functionality of how a user wants to
interact with a piece of software, user stories should not be thought of
as the same thing as features. Often when we think of the functionality
articulated as a feature it is a behavior that the software provides.
However, the description of a feature does not include how a user, uses
or invokes the feature. User stories keep the focus on the delivery of
value to users and often involve multiple features of a piece of
software.

Is a Story as Identification of Some Work (Work Item)?
======================================================

The simple answer is yes. A story represents work that needs to be
accomplished. The story is scheduled and tracked in a plan. However, a
story generally breaks down into a set of tasks that result in the
implementation of a story. This makes a story useful for planning
purposes. It represents a set of detailed tasks that should specified
close to when the story is scheduled to be implemented.

What is the difference between an Epic and a Story?
===================================================

Large Stories are called Epics. An Epic is a story that cannot be
implemented within one iteration so it must be broken down into finer
grained stories. This creates a natural hierarchy of requirements. Epics
are useful for course grained planning and scoping, but are not intended
to be detailed.

Many traditional requirements techniques have concepts of levels of
requirements and/or requirements decomposition (breaking requirements
into fine and finer levels of detail until they can be implemented.

In essence a Story and an Epic are the same. It's just one is at a level
of detail that it can be implemented within an iteration (Sprint).
Making the distinction has nominal value in general other that it is an
immediate expectation that an Epic needs to be decomposed before it can
be scheduled for implementation.

It is important to note that if/when an Epic is decomposed, it must be
fully decomposed. This means that the Epic must be broken into 2 or more
Stories that encompass all the behavior of the Epic. Keeping the Epic is
useful for understanding how the stories were created and/or derived.
However, the Epic should no longer useful for planning as the stories
now have all the details.

What if one Story has Similar steps to another Story?
=====================================================

Multiple Stories often similar steps. This is OK and expected. Per
above, the Story represents a requirement. If a user can interact with
the software and achieve the desired result per the user story then the
requirements has been fulfilled. Because a story can be a robust set of
interaction with the software, it is to be expected that other stories
that are similar would have a similar set of steps.

This is actually a gained efficiency. Stories implemented earlier in a
development lifecycle often take more time than similar stories later in
the development lifecycle. This is because some steps have already been
implemented or only needs slight modifications to accommodate the new
story.

Does a User Story Always Involve a User?
========================================

Technically yes and that user is a person. However, I find it more
useful to consider the "user" an Actor. This allows the entity
interacting with the software to be a person, another systems, a piece
of software, etc. If something wants/needs to interact with the software
under development, then there is a requirement that needs to be captured
to handle the interaction.

A Placeholder for a Future Conversation
=======================================

This is a powerful concept for User Stories. The idea, as you might
already see, is that a simple user story statement is not very complex
or robust -- it is rather simplistic. There is really not practical way
for a developer to be successful if they only had the simple user story
statement. What the statement does is allow for a placeholder to be
articulated with the understanding that more detail must be forthcoming.

The fleshing out of the story details often comes during the iteration
in which it is being implemented, if simple enough, or right before the
iteration during which the story will be implemented. This allows the
project to focus on stories at the right time. It is more efficient to
spend effort on near term work that has been prioritized to deliver the
most value sooner than later. If a story will not be implemented till
later, then don't spend time focusing on it's details when other more
important stories need attention.

However, story details and key decisions must be captured. There are
very few development efforts today that really don't need to capture
important details. Some of these will naturally be captured in the
Acceptance Criteria, but others, like non-functional requirements may
need to be captured in a Description or Story Elaboration section.
