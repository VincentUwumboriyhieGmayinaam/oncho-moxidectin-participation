# Data

The survey data are **not** in this repository. They contain household locations and community names that could
identify respondents. De-identified analysis files (`paper3_uptake.csv`) can be requested from the corresponding author
[FILL: name, email], subject to approval by [FILL: ethics committee] and the Twifo Atti-Morkwa District Health
Directorate. Household coordinates are not shared under any circumstances.

Place the files here before running `run_all.sh`:

```
data/
  paper3_uptake.csv
  community_codes.json
```

`community_codes.json` maps community names to the codes C01-C22 used in the papers. It is held by the study team and
the district health directorate and is shared only with approved users. Set the environment variable `COMMUNITY_KEY`
if it is stored elsewhere.

See `docs/codebook.md` for variable definitions.
