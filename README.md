# LeWM Coordinate Game

A JEPA-based world model (LeWorldModel) controlling an agent in a 2D mine-filled grid, using Model Predictive Control with CEM planning in latent space.

Based on: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture" (Maes, Le Lidec et al., 2026)

## Example Output

```
                                         ·                                         
                                        X·                             X   X       
                    X            X       ·                                  X      
                X  XX      X             · X   X         X                         
        ○!○○○○○                       X  ·       X                                 
         X    ○                          ·                                 X       
             X○                          ·         X      X                        
              ○     X    X       X       ·   X                    X          X     
 X      X     ○○○X                       X                    X                    
  X             ○○○○○                    X          X                X   X         
                X   ○○○○   X             ·               X                X        
          X    X      X○○○○XX      X  X  ·                                         
                          ○○○     XX   X X                             X           
      X          X          ○○           ·  X             X                        
                             ○○○     X   ·                                         
       X             X         ○○X  X    X                                         
    X                           ○       X·                         X               
                             X  ○        ·                    X    XX              
                       X  X   X ○○○      ·   X X            X                      
                                  ○○     ·          X            X                 
    X          X        X          ○○○   ·         X                               
                        X            ○○○○·  X                         X            
                X                       ○○○!○                              X       
 X                    X        X  X      X X○○○○      X                       X    
                                         ·     ○○   X       X      X               
                     X                   ·      ○                                  
       X                  X           X  X      ○   X     X     X                  
      X         X  X         X X         ·  X   ○                                  
        X      X  X            X      X  ·  X   ○         X           X      X     
       X     X    X X                    ·      ○                                  
                          X         X    X      ○○                            X    
                                         ·  X    ○○!○                              
                                         ·          ○                        X     
                 X                       ·          ○       X       X           X  
                              X          ·         X○                            X 
                      X               X  ·        X ○○  X                          
        X                 X X            ·           ○○○○       X          X   X   
                    X       X            ·              ○○○          X             
                                         ·        X     X ○                        
   X                                 X   ·                ○                        
                                         ·                ○○○                      
············X······X········································○○············X··X·····
               X                         ·               X   ○○○               X   
                        X      X         ·      X              ○                 X 
             XX  X                       ·       X   X XX      ○○                  
       X                                 ·                     X○                  
    X                          X X    X  ·           X          !○○              X 
              X                   X      ·          X             ○○               
                            X            ·                X    X   ○○○X      X X   
                             X      X    ·                           ○○ XX         
          X                      X X     ·                     X      ○            
                                         ·       X         XX         ○         X  
                                   X     · X            X             ○X   X       
                                X        ·     X                  X  ○○★     X     
                                 X       ·                           ○             
    X                                    · X                                       
 X   X       X   X        X              ·                              X          
          X                         X    ·                       X       X         
   X                           X         ·        X      X             X           
    X                                    X                            X     X      
   X    X                      X       X ·     X                                   
                             X     X     ·                                     X   
       X                                 · X                                    X  
      X             X          X         · X                   XX                  
                                 X     X X X                            X          
                      X         X        ·                      X                  
                                         ·           XX                            
                X                        ·       X                                 
     X                                   ·                              X      X   
             X    X                      ·                             X           
                  X                      ·                     X         X     X   
            X   X      X                 ·                                   X     
                X              XX   X    ·        X          X    X                
      XXX                                ·               X                         
                            XX           ·   X         X                           
 XX  X                                   ·      X                      X           
           X                             ·                          X           X  
                         X               ·     X                 X                 
                       X    X        X   ·                       X      X      X   
                   X                     ·                                         
                         X         X     ·                     X               X   
                  X                      ·            XX                     X     
                                         ·                                         
                                 
```

Legend: `★` agent at goal, `○` path taken, `X` mine, `!` mine hit, `·` axis

> **Note:** The agent does not currently avoid mines. The CEM planner only minimizes latent distance to the goal — it has no penalty for mine collisions. Mine avoidance is a planned improvement.

## How It Works

1. An **Encoder** maps the 10D state (x, y + 8 mine sensors) to a 64D latent embedding, with a BatchNorm projection head
2. A **Predictor** with residual connection predicts the next latent: `z_{t+1} = z_t + f(z_t, a_t)`
3. Training loss: `MSE(z_pred, z_target) + λ * SIGReg(Z)` — SIGReg prevents representation collapse by pushing latents toward an isotropic Gaussian distribution
4. **CEM planner** samples action sequences, rolls them out in latent space, and refines toward sequences that minimize latent distance to the goal
5. **MPC loop**: encode current state, plan via CEM, execute first action, replan

## Coordinate Game

- Agent trains on a 41x41 grid (±20) with 80 mines
- Agent plays on an 81x81 grid (±40) with 320 mines — testing out-of-distribution generalization
- 8 directional mine sensors (1-step visibility)
- Stepping on a mine = -1 penalty
- Goal: reach a random target position

## Files

| File | Description |
|------|-------------|
| `mpc_controller.py` | LeWorldModel (Encoder, Predictor, SIGReg), CEM planner, MPC controller |
| `coordinate_game.py` | Grid game environment, data collection, ASCII rendering, main loop |

## Quick Start

```bash
pip install -r requirements.txt
python coordinate_game.py
```
