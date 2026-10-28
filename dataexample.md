
--- nfl-big-data-bowl-2027\combine_results.csv ---
   draft_year  nfl_id combine_position  combine_height  combine_weight  \
0        2023   56040               TE          77.875             245   
1        2023   56010               DB          70.625             191   
2        2023   56085               TE          79.375             255   
3        2023   56056               WR          72.250             192   
4        2023   56118               WR          71.625             191   

   hand_size  arm_length  wing_span  ten_yd_split  forty  vertical  \
0     10.000      32.250     79.750          1.60   4.84      38.5   
1      8.750      29.625     72.250          1.58   4.59      33.5   
2     10.250      34.000     83.125          1.57   4.55      40.0   
3      9.625      31.875     76.125          1.51   4.33       NaN   
4      9.500      31.000     75.000          1.52   4.54      38.5   

   broad_jump  three_cone  short_shuttle  bench_reps  ngs_athleticism_score  \
0       125.0         NaN            NaN         NaN                     60   
1       116.0         NaN            NaN        23.0                     60   
2       128.0        6.87           4.12        23.0                     97   
3         NaN         NaN            NaN         NaN                     76   
4       120.0        6.98           4.15        14.0                     65   

   ngs_college_production_score  ngs_final_score  
0                           NaN              NaN  
1                          77.0             74.0  
2                          65.0             79.0  
3                          68.0             73.0  
4                          65.0             64.0  

--- nfl-big-data-bowl-2027\combine_tracking.csv ---
   draft_year                          event_id  nfl_id entity_type  \
0        2025  20250302100900_1740941802777_000   58286      PLAYER   
1        2025  20250302100900_1740941802777_000   58286      PLAYER   
2        2025  20250302100900_1740941802777_000   58286      PLAYER   
3        2025  20250302100900_1740941802777_000   58286      PLAYER   
4        2025  20250302100900_1740941802777_000   58286      PLAYER   

                      time       drill_type                      drill_name  \
0  2025-03-02T18:56:38.200  SKILL_DRILLS_OL  OL_PULL_DRILL_FOLD_BLOCK_RIGHT   
1  2025-03-02T18:56:38.300  SKILL_DRILLS_OL  OL_PULL_DRILL_FOLD_BLOCK_RIGHT   
2  2025-03-02T18:56:38.400  SKILL_DRILLS_OL  OL_PULL_DRILL_FOLD_BLOCK_RIGHT   
3  2025-03-02T18:56:38.500  SKILL_DRILLS_OL  OL_PULL_DRILL_FOLD_BLOCK_RIGHT   
4  2025-03-02T18:56:38.600  SKILL_DRILLS_OL  OL_PULL_DRILL_FOLD_BLOCK_RIGHT   

   attempt          x          y         s         a       dis         dir  
0        4  44.785924  28.285528  0.220828  0.682443  0.011085  250.278576  
1        4  44.723780  28.297391  0.332931  0.619551  0.003864  257.319206  
2        4  44.632680  28.279528  0.481838  0.596455  0.008305  253.176813  
3        4  44.544022  28.243767  0.599350  0.664436  0.007964  247.044153  
4        4  44.471466  28.210294  0.657540  0.908250  0.005382  240.622668  

--- nfl-big-data-bowl-2027\game_tracking_2023.csv ---
      game_id  play_id  nfl_id                     time      x      y    s  \
0  2024010707      570   56450  2024-01-07T21:46:07.700  42.17  25.29  0.0   
1  2024010707      570   56450  2024-01-07T21:46:07.800  42.17  25.29  0.0   
2  2024010707      570   56450  2024-01-07T21:46:07.900  42.18  25.29  0.0   
3  2024010707      570   56450  2024-01-07T21:46:08.000  42.18  25.29  0.0   
4  2024010707      570   56450  2024-01-07T21:46:08.100  42.18  25.29  0.0   

     a  dis       o     dir event  
0  0.0  0.0  290.97  213.26   NaN  
1  0.0  0.0  290.97  210.36   NaN  
2  0.0  0.0  290.97  205.45   NaN  
3  0.0  0.0  290.97  200.73   NaN  
4  0.0  0.0  290.97  199.27   NaN  

--- nfl-big-data-bowl-2027\game_tracking_2024.csv ---
      game_id  play_id  nfl_id                     time      x      y     s  \
0  2024080952     3379   57367  2024-08-10T01:30:31.800  87.09  22.52  0.28   
1  2024080952     3379   57367  2024-08-10T01:30:31.900  87.11  22.51  0.24   
2  2024080952     3379   57367  2024-08-10T01:30:32.000  87.12  22.50  0.17   
3  2024080952     3379   57367  2024-08-10T01:30:32.100  87.13  22.49  0.15   
4  2024080952     3379   57367  2024-08-10T01:30:32.200  87.14  22.48  0.11   

      a   dis      o     dir event  
0  0.11  0.03  98.51  126.08   NaN  
1  0.18  0.02  97.33  125.81   NaN  
2  0.24  0.01  96.65  124.99   NaN  
3  0.21  0.02  95.42  120.75   NaN  
4  0.19  0.01  95.42  136.32   NaN  

--- nfl-big-data-bowl-2027\game_tracking_2025.csv ---
      game_id  play_id  nfl_id                     time      x      y     s  \
0  2025090701      290   57347  2025-09-07T17:09:39.500  83.06  32.06  0.10   
1  2025090701      290   57347  2025-09-07T17:09:39.600  83.07  32.07  0.10   
2  2025090701      290   57347  2025-09-07T17:09:39.700  83.06  32.06  0.07   
3  2025090701      290   57347  2025-09-07T17:09:39.800  83.07  32.07  0.07   
4  2025090701      290   57347  2025-09-07T17:09:39.900  83.08  32.07  0.06   

      a   dis       o    dir event  
0  0.06  0.04    8.00  18.86   NaN  
1  0.07  0.02    2.61  21.95   NaN  
2  0.05  0.02  357.42  16.24   NaN  
3  0.05  0.01  352.73  18.93   NaN  
4  0.05  0.01  349.52  20.80   NaN  

--- nfl-big-data-bowl-2027\games.csv ---
      game_id  game_key  season season_type  week home_team_abbr  \
0  2023090700     59173    2023         REG     1             KC   
1  2023091000     59174    2023         REG     1            ATL   
2  2023091001     59175    2023         REG     1            BAL   
3  2023091002     59176    2023         REG     1            CLE   
4  2023091003     59177    2023         REG     1            IND   

  visitor_team_abbr  
0               DET  
1               CAR  
2               HOU  
3               CIN  
4               JAX  

--- nfl-big-data-bowl-2027\player_career_successes.csv ---
   nfl_id  career_offensive_snaps  career_defensive_snaps  \
0   55867                       0                    1884   
1   55870                    2802                       0   
2   55874                    3222                       0   
3   55875                    3011                       0   
4   55877                       0                    1056   

   career_special_teams_snaps  career_games_active  career_games_started  \
0                          92                   51                    44   
1                         169                   47                    43   
2                         167                   51                    49   
3                         170                   51                    48   
4                         387                   51                     2   

   ap_all_pro_1st_team  ap_all_pro_2nd_team  pro_bowl_original_ballot  
0                    1                    0                         1  
1                    0                    0                         0  
2                    0                    1                         0  
3                    0                    0                         0  
4                    0                    0                         0  

--- nfl-big-data-bowl-2027\player_play.csv ---
      game_id  play_id team_abbr  nfl_id lined_up_position  \
0  2025111610     2279       BAL   57183                 T   
1  2026011701     1626        SF   58248     INTERIOR_LINE   
2  2025101900     2838       LAR   57140              EDGE   
3  2025111300     2628       NYJ   58209                 T   
4  2025121406     2653       NYG   57191               OLB   

                                    play_description  quarter  down  \
0  (8:42) (Shotgun) L.Jackson pass incomplete dee...        3     1   
1  (3:48) (Shotgun) K.Walker left end to SEA 28 f...        2     2   
2  (7:01) C.Van Lanen reported in as eligible.  T...        3     2   
3  (4:05) I.Williams left end to NE 22 for 4 yard...        3     1   
4  (10:08) (No Huddle, Shotgun) M.Mariota scrambl...        3     2   

   yards_to_go possession_team  ... sack  tackle assist  caused_forced_fumble  \
0           10             BAL  ...  NaN     NaN    NaN                   NaN   
1            7             SEA  ...  NaN     NaN    NaN                   NaN   
2            2             JAX  ...  0.0     NaN    NaN                   NaN   
3           10             NYJ  ...  NaN     NaN    NaN                   NaN   
4           10             WAS  ...  0.0     1.0    0.0                   0.0   

   recovered_fumble tackle_for_loss  time_to_pressure  time_to_qb_hurry  \
0               NaN             NaN               NaN               NaN   
1               NaN             NaN               NaN               NaN   
2               NaN             NaN               NaN               NaN   
3               NaN             NaN               NaN               NaN   
4               NaN           False               NaN               NaN   

   coverage_assignment  coverage_assignment_at_snap  
0                  NaN                          NaN  
1                  NaN                          NaN  
2        NO_ASSIGNMENT                NO_ASSIGNMENT  
3                  NaN                          NaN  
4       HOOK_CURL_LEFT               CURL_FLAT_LEFT  

[5 rows x 64 columns]

--- nfl-big-data-bowl-2027\players.csv ---
   nfl_id     display_name  draft_year nfl_position  birth_date  college_name  \
0   55867    Will Anderson        2023           DE  2001-09-02       Alabama   
1   55870    Paris Johnson        2023            T  2001-07-31    Ohio State   
2   55874   Darnell Wright        2023            T  2001-08-10     Tennessee   
3   55875  Peter Skoronski        2023            G  2001-07-31  Northwestern   
4   55877   Lukas Van Ness        2023           DE  2001-07-06          Iowa   

        college_conference  draft_round  draft_pick_within_round  \
0  Southeastern Conference          1.0                      3.0   
1       Big Ten Conference          1.0                      6.0   
2  Southeastern Conference          1.0                     10.0   
3       Big Ten Conference          1.0                     11.0   
4       Big Ten Conference          1.0                     13.0   

   draft_overall_pick  
0                 3.0  
1                 6.0  
2                10.0  
3                11.0  
4                13.0  
