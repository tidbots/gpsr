# General Purpose Service Robot
The robot is asked to understand and execute commands requiring a wide range of different
abilities.

**Main Goal:** Execute three commands issued by the operator.

## Focus
Task planning, object/people detection and recognition, object feature recognition, object manipulation

## Setup
- Locations:
   - **Task location:** The task takes place inside the Arena. Commands may require the
robot to leave the room. The Arena is in its nominal configuration for this task.
   - **Start location:** The robot starts outside the Arena. When the door opens, it must
navigate towards the Instruction Point.
   - **Instruction point:** The robot returns to this point after completing all the commands.

- People:
  - *Professional Operator:* The referee issues standard commands to the robot. If the
robot consistently fails to understand the command (e.g. after three tries), teams can
use a custom operator.

##Procedure
1. **Instruction point:** At least two hours before the test, the referees announce the location
of the Instruction Point.

2. **Command execution:** The robot will decide how the commands will be issued and
advise the operator, i.e., either consecutively or one-by-one. If the commands are issued
one-by-one, the robot must return to the operator after completing each task.

3. **Back to the instruction point:** The robot goes back to the Instruction Point after
completing all the commands given by the operator.

4. **Pausing the Timer:** The referee pauses the timer as soon as the robot reaches the
instruction point to reset the arena for the next command. The timer resumes once the
referee signals the start of the next command.

## Additional Rules and Remarks
1. **Interleaved Task Bonus:** The robot receives an additional bonus if it successfully completes
commands in an interleaved order rather than strictly consecutively. This bonus is
awarded only when all three commands are received at once. The interleaved execution must be meaningful, for example by saving time or reducing unnecessary movements.

Example:<br>
The robot first picks up an object, then performs another task along the way, and
only afterward delivers the object to its original destination.

2. **Partial Scoring:** The solution allows partial scoring.

3. **Command generator:** Tasks will be generated using the official command generator1.
Once a command has been generated it will be entered into an LLM to re-generate a
similar phrase, e.g. the generated command is ”get me a coke from the kitchen” rephrased
command is ”Go to the kitchen, find a coke, and bring it to me”. Each command
may be re-phrased up to 3 times getting simpler with each rephrasing.

https://github.com/RoboCupAtHome/CommandGenerator

4. **Test start:** The robot moves to the Instruction Point when the arena door is open.

5. **Team Coaching:** Teams are not allowed to coach, or instruct the operators. Doing so
results in disqualification from the task.

6. **Custom Operators:** If a custom operator is used they can only choose between the three
re-phrased commands to give.

7. **Autonomy Skip:** Score reductions apply in the following cases:
- Use of a custom operator.
- Bypassing speech recognition.
- Receiving human assistance to accomplish a command.
- Instructing a human assistant to perform the whole task.
- QR codes will not be available.

## Referee Instructions
- Provide the commands to the operators.

## OC Instructions
At least two hours before the test:<br>
- Generate the robot commands and pass through LLM to get a similar command (do not
reveal them to the teams).
- Announce the location of the instruction point.
- Recruit volunteers to assist during the test.

During the test:<br>
- The arena will be setup for all command executions.

## Score Sheet
The maximum time for this test is **7:00** minutes.




| Action  | Score |
| ------------- | ------------- |
| Main Goal  |   |
| Understand the spoken command  | 3×80  |
| Demonstrate a plan has been generated  | 3×100 |
| Solving the command  | 3×250  |
| Bonus Rewards  |   |
| Interleaved Task Bonus  | 200  |
| Penalties  |   |
| Using a custom operator  | 3×-20  |
| Request a rephrasing  | 6×-30  |
| Bypassing speech recognition  | 3×-50  |
| Human assistance: will apply a percentage penalty according to similar penalties in
other tests.  | 3×-250  |
| Special Penalties & Bonuses  |   |
| Not attending (see sec. 3.8.1)  | -500  |
| Using alternative start signal (see sec. 3.6.8)  | -100  |
| Outstanding performance (see sec. 3.8.3)  | 149  |
| ------------- | ------------- |
Total Score (excluding special penalties & standard bonuses) 1490
