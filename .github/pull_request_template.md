# Pull Request


[JIRA Issue](put-jira-url-here)

Start your commit messages with one of the following keywords to trigger version updates:
- `patch: `: patch version update (1.0.0 -> 1.0.1)
- `minor: `: Minor version update (1.0.0 -> 1.1.0)
- `major: `: Major version update, breaking changes (1.0.0 -> 2.0.0)

## Description
Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context.

Fixes # (issue)

## Changes
- 

## How to Trigger Workflows
First you'll need to add the relevant Docker metadata to the `.github/configs/experiments.yml` file; you'll need to add the `working_directory` (i.e the directory where the experiment's `Dockerfile` is located) and the full ECR repostitory name (`ecr_repository`)

You can trigger the various Github workflows by commenting the following commands on the PR:
- `docker push [experiment_name] [environment]`
- `publish [environment]` to publish the `signature-dojo` package


Where `[experiment_name]` is the name of the experiment defined in `.github/configs/experiments.yml`, and `[environment]` is `development`, `staging`, or `production`

After you comment, Github will push a comment to the PR with a link to the running workflow (10 seconds)

**Note**: Deployments to `production` will only work if the PR has been approved and if the deployment to `staging` was successful.


