# BPTT and world-model design

## Preserved task

Real collection and evaluation use the frozen ConTrack task source. Actor observations, Gaussian Actor structure, task/style value heads, residual action mapping, reference frame semantics, real reward terms/weights and success thresholds are preserved. In this BPTT implementation, collection uses the original initial Gaussian standard deviation 1.0 as a fixed value; PPO retains its trainable standard deviation.

The original task has a velocity-cache advanced-indexing quirk. The differentiable task deliberately matches the actual released acceleration reward rather than silently repairing that quirk. Reward-parity tests execute both upstream embodiments to protect this behavior.

## WM input and output

For each clip, one joint world model takes:

- Physical state: all controlled/extra joint positions and velocities, plus each object's position, wxyz quaternion, linear velocity and angular velocity.
- Executed joint targets after the original residual action mapping.
- Training-only physical context, including masses, inertia, friction and actuator parameters.

It predicts physical increments (SO(3) increments for orientation), per-hand/link/object contact logits and object-local sensor contact points. It recursively reconstructs the next physical state. Sensor points are a ConTrack contact-distance reward requirement; they are an adaptation beyond the Dex reference's physical-state/contact design.

The included [Hammer descriptor](../results/hammer_best/descriptor.json) has two hands, 38 controlled joints, one object and 238 context features. Thus the physical state has 89 features, joint targets have 38, WM input has 365, and output has 148 (88 physical increments + 20 contact logits + 60 contact-point coordinates). These dimensions differ from the single-hand example and from Dex's robot.

WM fitting recursively supervises eight steps, beginning at a real replay state and reusing predictions with recorded executed targets. Training/holdout environments are separate and sequence masks stop at episode boundaries. Targets never become recurrent inputs inside a predicted sequence.

## Actor BPTT

The Actor unrolls 32 mean-action steps through the WM. WM and Critic parameters are frozen during Actor backward while gradients with respect to their inputs remain enabled. Intermediate state, control and observation recurrence retain gradients across the whole unroll.

The objective mixes the ConTrack task/style channels using the dual-controller weight, sums discounted predicted rewards with survival weights, and adds terminal Critic bootstrap. Soft contact rewards and analytic soft pose continuation make imagination differentiable; the Actor's contact observation uses hard values with straight-through gradients. **These are training proxies; real environment rewards and final evaluation use upstream semantics.**

Reference timeouts bootstrap from the terminal physical observation and end that imagined episode. Critic fitting follows Actor backward and uses twin lambda returns with gamma 0.99 and lambda 0.95. Only detached trajectory copies are used for Critic fitting; they do not truncate Actor BPTT.

The donor-style optimizer uses AdamW, global gradient norm clipping at 1.0 followed by Actor gradient value clipping at ±0.1. WM/Actor batches are 2048/1024, with microbatches 512/256 and shared full-batch denominators. AMP, TF32 and `torch.compile` are enabled in the winning configuration. This release does not include later CUDA Graph or objective experiments.

## Winning schedule

The exact [configuration](../configs/hammer_best.json) uses preset e3, seed 2026, learning rate 0.0002, a three-layer 2048-wide WM, 64 initial real vector steps, 500 WM-only updates, and 10 WM minibatches per update. Warmup UTD is 0.4; normal UTD is 0.0625.

`budget_updates=500` denotes 500 PPO-equivalent iterations, each 16 real vector steps. Including initial collection/warmup, training ends at 8000 real vector steps × 8000 environments = 64 million real transitions. The recorded run completed 418 Actor updates; 500 is not the number of Actor BPTT updates.

## Sources

- ConTrack: `f5efe18e4b1f986566b252265c4bf63f3b67c6c6`.
- Dex-Supermimic method reference: `2becb7380fed86171056f9bf4338b2a6a0c14b01`.
- Winning extension: `a232096726ae6c9a6523aabe3b8cd7a156bd66a0`.

The extension is independently implemented. This is an adaptation of the donor's training design to ConTrack, not a numerical comparison against donor weights or a claim of a new BPTT algorithm.
