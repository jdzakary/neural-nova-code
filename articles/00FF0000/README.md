# Training a *(completely)* Unbeatable Tic-Tac-Toe AI using Reinforcement Learning

I found this while looking for information on hyperparameter optimization. The article ended with a challenge to make a perfect 'O' agent. I remembered a trick I used for a PFSP setup to do something like that, so I thought I'd try it here.

I also restructured things to make it a little more cross-compatible for people starting it up, and updated the training script to be compatible with the latest RLlib.

## My results for the top 5 trials, when replicating the original result, were as follows:

Trials:

```
0  b150e5c8  TERMINATED 0 days 00:16:01  0.997  0.003  0.000     21
1  a294744a  TERMINATED 0 days 00:20:26  0.997  0.003  0.000     27
2  2da66817  TERMINATED 0 days 00:37:20  0.995  0.000  0.005     50 # No WinX?
3  af610087  TERMINATED 0 days 00:16:46  0.992  0.003  0.005     21
4  ca0eceae  TERMINATED 0 days 01:29:16  0.957  0.019  0.024    117
5  8242f238  TERMINATED 0 days 01:12:49  0.828  0.098  0.074     96
6  2dd2537e  TERMINATED 0 days 01:28:36  0.826  0.135  0.039    116
7  095eb04c  TERMINATED 0 days 00:35:02  0.712  0.172  0.116     45
8  98cc34f1  TERMINATED 0 days 01:07:55  0.593  0.291  0.117     88
```

```
Evidently not. It does have one fewer failure case, though:

Model is O:
Games Played: 495
Model Loses: 2
Win X: 0.0040
Win O: 0.6970
Tie  : 0.2990
[0, 4, 8, 6, 2, 1, 5]
[8, 4, 0, 6, 2, 1, 5]

Model is X:
Games Played: 200
Model Loses: 0
Win X: 0.6800
Win O: 0.0000
Tie  : 0.3200
```

```
+==========+====================+=======+===========+========================+======================+====================+
| trial    | Metric             | Iters | Best_Iter | lr                     | tie_penalty          | gamma              |
+==========+====================+=======+===========+========================+======================+====================+
| b150e5c8 | 0.9971740993314576 | 21    | 21        | 0.00028680243649277786 | 0.694662071094744    | 0.984343331592276  |
| a294744a | 0.9969647080737128 | 27    | 27        | 6.563820049480557e-05  | 0.3351991760505908   | 0.9198887413305485 |
| 2da66817 | 0.994607515190224  | 50    | 50        | 9.16829806363971e-05   | 0.03174324356754088  | 0.916639034876105  |
| af610087 | 0.9917480374002904 | 21    | 21        | 0.00012249873224163968 | 0.3584575210703531   | 0.954354300759849  |
| ca0eceae | 0.957779324053022  | 117   | 102       | 3.234996176722559e-05  | -0.7172583647875148  | 0.9539707008678788 |
| 2dd2537e | 0.9086573735549264 | 116   | 97        | 0.0020969750984383543  | -0.12362702207911913 | 0.9356379087689773 |
| 8242f238 | 0.871434971913041  | 96    | 43        | 0.0016641559256606873  | -0.19496502955576323 | 0.9551933705407369 |
| 98cc34f1 | 0.7267810556994212 | 88    | 48        | 0.002364565331713159   | -0.7004709335628887  | 0.9476879582618583 |
| 095eb04c | 0.7118417033109606 | 45    | 45        | 0.003476168392400587   | -0.6379072401826693  | 0.840285356363364  |
| 9004554f | 0.1280261124229172 | 1     | 1         | 4.7973917731170575e-05 | -0.954755269690678   | 0.9777311400807173 |
+----------+--------------------+-------+-----------+------------------------+----------------------+--------------------+

```

## My results, when running my solution, were as follows:

```
0  31b7349c  RUNNING 0 days 01:29:14  0.979  0.019  0.001    117
1  6a00f222  RUNNING 0 days 01:28:25  0.902  0.090  0.008    116 # this one turned out best
2  bb97dc4c  RUNNING 0 days 01:28:52  0.851  0.025  0.124    117
3  ac419e2f  RUNNING 0 days 01:28:31  0.700  0.056  0.244    115
4  9b197b43  RUNNING 0 days 01:29:28  0.512  0.434  0.054    116
```

```
Model is O:
Games Played: 587
Model Loses: 0
Win X: 0.0000
Win O: 0.6559
Tie  : 0.3441

Model is X:
Games Played: 91
Model Loses: 0
Win X: 0.9560
Win O: 0.0000
Tie  : 0.0440
```

*There's what we're after!~

```
+==========+====================+=======+===========+========================+=====================+======================+====================+
| trial    | Metric             | Iters | Best_Iter | lr                     | x_tie_penalty       | o_tie_penalty        | gamma              |
+==========+====================+=======+===========+========================+=====================+======================+====================+
| bb97dc4c | 0.9999999999999944 | 117   | 100       | 0.00020071654256718154 | 0.8891175850956738  | -0.14344294993766604 | 0.8681381250099209 |
| 31b7349c | 0.9986867463942116 | 117   | 112       | 1.1537276533002902e-05 | 0.37838556253374667 | -0.20312599347084648 | 0.960783298104163  |
| 6a00f222 | 0.9785739682438944 | 116   | 86        | 6.290241910993141e-05  | 0.4110560456465997  | -0.42601942052495445 | 0.8228049023584615 |
| ac419e2f | 0.946309487409066  | 115   | 46        | 0.00418574432892795    | 0.7489793532062917  | -0.22283354722465032 | 0.9575636977794302 |
| 9b197b43 | 0.6231954637491537 | 116   | 41        | 0.006953192454916885   | 0.3692113879819525  | -0.9459224038321554  | 0.8880610654632924 |
+----------+--------------------+-------+-----------+------------------------+---------------------+----------------------+--------------------+
```

## Run Instructions

```
python training.py --x-lose-reward -.1 --o-win-reward .1 --stop-tie-threshold 1.0
python analysis.py --experiment-name 'Settings: 100 10 -10 -100 100' --create-plots "6a00f222"
python exporting.py --checkpoint-dir 'Settings: 100 10 -10 -100 100/6a00f222/checkpoint_000004' --export-dir 'att1'
python validation.py --export-dir 'att1'
```

**Postmortem:** It's certainly possible to solve this better than I did. I'd expect hard-coding X's tie penalty to something resembling -1, while hard-coding O's tie penalty to something resembling 1, should do the trick more reliably. Likewise, setting agents to not terminate, rather than using .999 as the stopping value, probably does more harm than good in the expected case. Nonetheless, training a dedicated exploiter agent (that is, an agent that only cares about winning, and has no interest in playing conservatively) as X will demonstrably get you an O that plays perfectly. 