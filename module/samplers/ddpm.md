## Noising Process
##### The scaled linear schedule used during training of the Stable Diffusion 1.x series.

$$
{{\bf{x}}_t} = \sqrt {1 - {\beta _t}} {{\bf{x}}_{t - 1}} + \sqrt {{\beta _t}} {{\bf{\varepsilon }}_{t - 1}},\qquad {{\bf{\varepsilon }}_{t - 1}}\sim{\cal N}\left( {0,{\bf{I}}} \right),\qquad {\beta _t} = {\left( {\sqrt {{\beta _{{\rm{start}}}}}  + \frac{t}{{T - 1}}\left( {\sqrt {{\beta _{{\rm{end}}}}}  - \sqrt {{\beta _{{\rm{start}}}}} } \right)} \right)^2},\qquad \left\{ {\begin{array}{*{20}{l}}
{T = {\rm{1000}}}\\
{{\beta _{{\rm{start}}}} = 0.00085}\\
{{\beta _{{\rm{end}}}} = 0.01200}
\end{array}} \right.
$$

##### For convenience of derivation, replace beta with alpha.
$$
{{\bf{x}}_t} = \sqrt {{\alpha _t}} {{\bf{x}}_{t - 1}} + \sqrt {1 - {\alpha _t}} {{\bf{\varepsilon }}_{t - 1}},\qquad {\alpha _t} = 1 - {\beta _t}
$$

##### Substitute ${\bf{x}}_{t - 1}$ and apply the linearity of the normal distribution:
$$
{{\bf{x}}_t} = \sqrt {{\alpha _t}{\alpha _{t - 1}}} {{\bf{x}}_{t - 2}} + \left( {\sqrt {{\alpha _t}\left( {1 - {\alpha _{t - 1}}} \right)} {{\bf{\varepsilon }}_{t - 2}} + \sqrt {1 - {\alpha _t}} {{\bf{\varepsilon }}_{t - 1}}} \right),\qquad \left\{ \begin{array}{l}
\sqrt {{\alpha _t}\left( {1 - {\alpha _{t - 1}}} \right)} {{\bf{\varepsilon }}_{t - 2}}\sim{\cal N}\left( {0,{\alpha _t}\left( {1 - {\alpha _{t - 1}}} \right)} \right)\\
\sqrt {1 - {\alpha _t}} {{\bf{\varepsilon }}_{t - 1}} \sim {\cal N}\left( {0,1 - {\alpha _t}} \right)\\
\left( {\sqrt {{\alpha _t}\left( {1 - {\alpha _{t - 1}}} \right)} {{\bf{\varepsilon }}_{t - 2}} + \sqrt {1 - {\alpha _t}} {{\bf{\varepsilon }}_{t - 1}}} \right) \sim {\cal N}\left( {0,1 - {\alpha _t}{\alpha _{t - 1}}} \right)
\end{array} \right.
$$

##### Simplify to the following form. Further expand to ${\bf{x}}_0$:
$$
{{\bf{x}}_t} = \sqrt {{\alpha _t}{\alpha _{t - 1}}} {{\bf{x}}_{t - 2}} + \sqrt {1 - {\alpha _t}{\alpha _{t - 1}}} {\bf{\varepsilon }} = \sqrt {{\alpha _t}{\alpha _{t - 1}} \cdots {\alpha _1}} {{\bf{x}}_0} + \sqrt {1 - {\alpha _t}{\alpha _{t - 1}} \cdots {\alpha _1}} {\bf{\varepsilon }},\qquad {\bf{\varepsilon }}\sim{\cal N}\left( {0,{\bf{I}}} \right)
$$

##### Simplify using $\bar{\alpha}_t$ to the following form:
$$
{{\bf{x}}_t} = \sqrt {{{\bar \alpha }_t}} {{\bf{x}}_0} + \sqrt {1 - {{\bar \alpha }_t}} {\bf{\varepsilon }},\qquad {{\bar \alpha }_t} = {\alpha _t}{\alpha _{t - 1}} \cdots {\alpha _1},\qquad {\bf{\varepsilon }}\sim{\cal N}\left( {0,{\bf{I}}} \right)
$$

---

## Denoising Process
##### The diffusion model assumes that the noising process is a Markov process. During denoising, given the image $\mathbf{x}_t$ at a certain timestep, we want to infer the state of the previous timestep $\mathbf{x}_{t-1}$, i.e., $P(\mathbf{x}_{t-1} | \mathbf{x}_t)$. By conditioning on $\mathbf{x}_0$ and applying Bayes' theorem, this can be rewritten as:
$$
P\left( {{{\bf{x}}_{t - 1}}|{{\bf{x}}_t}} \right) = P\left( {{{\bf{x}}_{t - 1}}|{{\bf{x}}_t},{{\bf{x}}_0}} \right) = \frac{{P\left( {{{\bf{x}}_t}|{{\bf{x}}_{t - 1}},{{\bf{x}}_0}} \right)P\left( {{{\bf{x}}_{t - 1}}|{{\bf{x}}_0}} \right)}}{{P\left( {{{\bf{x}}_t}|{{\bf{x}}_0}} \right)}}
$$

##### Using the noising process formula and the linearity of the normal distribution:
$$
\left\{ \begin{array}{l}
{{\bf{x}}_t} = \sqrt {{\alpha _t}} {{\bf{x}}_{t - 1}} + \sqrt {1 - {\alpha _t}} {{\bf{\varepsilon }}_{t - 1}} \Rightarrow P\left( {{{\bf{x}}_t}|{{\bf{x}}_{t - 1}},{{\bf{x}}_0}} \right)\sim{\cal N}\left( {\sqrt {{\alpha _t}} {{\bf{x}}_{t - 1}},1 - {\alpha _t}} \right)\\
{{\bf{x}}_{t - 1}} = \sqrt {{{\bar \alpha }_{t - 1}}} {{\bf{x}}_0} + \sqrt {1 - {{\bar \alpha }_{t - 1}}} {\bf{\varepsilon }} \Rightarrow P\left( {{{\bf{x}}_{t - 1}}|{{\bf{x}}_0}} \right)\sim{\cal N}\left( {\sqrt {{{\bar \alpha }_{t - 1}}} {{\bf{x}}_0},1 - {{\bar \alpha }_{t - 1}}} \right)\\
{{\bf{x}}_t} = \sqrt {{{\bar \alpha }_t}} {{\bf{x}}_0} + \sqrt {1 - {{\bar \alpha }_t}} {\bf{\varepsilon }} \Rightarrow P\left( {{{\bf{x}}_t}|{{\bf{x}}_0}} \right)\sim{\cal N}\left( {\sqrt {{{\bar \alpha }_t}} {{\bf{x}}_0},1 - {{\bar \alpha }_t}} \right)\\
{f_{\cal N}}(x;\mu ,{\sigma ^2}) = \frac{1}{{\sigma \sqrt {2\pi } }}\exp \left( { - \frac{{{{(x - \mu )}^2}}}{{2{\sigma ^2}}}} \right)
\end{array} \right.
$$

##### Rearrange into a proportional form (since we only care about $\mathbf{x}_{t-1}$, other parameters are treated as constants):
$$
\begin{array}{l}
P\left( {{{\bf{x}}_{t - 1}}|{{\bf{x}}_t}} \right) \propto \exp \left[ { - \frac{1}{2}\left( {\frac{{{{\left( {{{\bf{x}}_t} - \sqrt {{\alpha _t}} {{\bf{x}}_{t - 1}}} \right)}^2}}}{{1 - {\alpha _t}}} + \frac{{{{\left( {{{\bf{x}}_{t - 1}} - \sqrt {{{\bar \alpha }_{t - 1}}} {{\bf{x}}_0}} \right)}^2}}}{{1 - {{\bar \alpha }_{t - 1}}}} - \frac{{{{\left( {{{\bf{x}}_t} - \sqrt {{{\bar \alpha }_t}} {{\bf{x}}_0}} \right)}^2}}}{{1 - {{\bar \alpha }_t}}}} \right)} \right]\\
 = \exp \left[ { - \frac{1}{2}\left( {\left( {\frac{{{\alpha _t}}}{{1 - {\alpha _t}}} + \frac{1}{{1 - {{\bar \alpha }_{t - 1}}}}} \right){{\bf{x}}_{t - 1}}^2 - \left( {\frac{{2\sqrt {{\alpha _t}} }}{{1 - {\alpha _t}}}{{\bf{x}}_t} + \frac{{2\sqrt {{{\bar \alpha }_{t - 1}}} }}{{1 - {{\bar \alpha }_{t - 1}}}}{{\bf{x}}_0}} \right){{\bf{x}}_{t - 1}} + C\left( {{{\bf{x}}_t},{{\bf{x}}_0}} \right)} \right)} \right]
\end{array}
$$

##### By Bayes' theorem, $P(\mathbf{x}_{t-1} | \mathbf{x}_t)$ still follows a normal distribution. Let its mean and variance be $\mu$ and $\sigma^2$, respectively. We construct a probability density function for it:
$$
\exp \left[ { - \frac{1}{2}\left( {\frac{1}{{{\sigma ^2}}}{x^2} - \frac{{2\mu }}{{{\sigma ^2}}}x + \frac{{{\mu ^2}}}{{{\sigma ^2}}}} \right)} \right]
$$

##### By matching the corresponding parameters, we obtain the following system of equations:
$$
\left\{ \begin{array}{l}
\frac{{{\alpha _t}}}{{1 - {\alpha _t}}} + \frac{1}{{1 - {{\bar \alpha }_{t - 1}}}} = \frac{1}{{{\sigma ^2}}}\\
\frac{{2\sqrt {{\alpha _t}} }}{{1 - {\alpha _t}}}{{\bf{x}}_t} + \frac{{2\sqrt {{{\bar \alpha }_{t - 1}}} }}{{1 - {{\bar \alpha }_{t - 1}}}}{{\bf{x}}_0} = \frac{{2\mu }}{{{\sigma ^2}}}
\end{array} \right.
$$

##### Substituting $\mathbf{x}_0$ using the formula derived in the noising process, we solve for the mean $\mu$ and variance $\sigma^2$:
$$
\begin{array}{l}
{\sigma ^2} = \frac{{\left( {1 - {\alpha _t}} \right)\left( {1 - {{\bar \alpha }_{t - 1}}} \right)}}{{1 - {{\bar \alpha }_t}}}\\
\mu  = \frac{{\left( {1 - {{\bar \alpha }_{t - 1}}} \right)\sqrt {{\alpha _t}} }}{{1 - {{\bar \alpha }_t}}}{{\bf{x}}_t} + \frac{{\left( {1 - {\alpha _t}} \right)\sqrt {{{\bar \alpha }_{t - 1}}} }}{{1 - {{\bar \alpha }_t}}}{{\bf{x}}_0} = \frac{1}{{\sqrt {{\alpha _t}} }}\left( {{{\bf{x}}_t} - \frac{{1 - {\alpha _t}}}{{\sqrt {1 - {{\bar \alpha }_t}} }}{\bf{\varepsilon }}} \right)
\end{array}
$$

##### Thus, $P(\mathbf{x}_{t-1} | \mathbf{x}_t)$ follows the normal distribution below, from which we can sample to obtain $\mathbf{x}_{t-1}$:
$$
P\left( {{{\bf{x}}_{t - 1}}|{{\bf{x}}_t}} \right)\sim{\cal N}\left( {\frac{1}{{\sqrt {{\alpha _t}} }}\left( {{{\bf{x}}_t} - \frac{{1 - {\alpha _t}}}{{\sqrt {1 - {{\bar \alpha }_t}} }}{\bf{\varepsilon }}} \right),\frac{{\left( {1 - {\alpha _t}} \right)\left( {1 - {{\bar \alpha }_{t - 1}}} \right)}}{{1 - {{\bar \alpha }_t}}}} \right)
$$