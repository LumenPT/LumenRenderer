library(
    identifier: 'JenkinsPipelineUtilities@master',
    retriever: modernSCM([$class: 'GitSCMSource', credentialsId: '', remote: 'https://github.com/timrademaker/JenkinsPipelineUtilities.git']))

pipeline {

    parameters{

        //GitHub
        credentials(    credentialType: 'Username with password',
                        defaultValue: params.GITHUB_CREDENTIALS ? params.GITHUB_CREDENTIALS : '',
                        description: 'The credentials to use to login to github in order to be able to pull the project source code',
                        name: 'GITHUB_CREDENTIALS',
                        required: true)

        string(         defaultValue: params.GITHUB_URL ? params.GITHUB_URL : '',
                        description: 'The URL to the GitHub repository to use for the project source',
                        name: 'GITHUB_URL')

        string(         defaultValue: params.GITHUB_BRANCH ? params.GITHUB_BRANCH : 'master',
                        description: 'The branch to pull from the GitHub repository to use for the project source',
                        name: 'GITHUB_BRANCH')

        //Build
        string(         defaultValue: params.BUILD_SRC_DIR ? params.BUILD_SRC_DIR : '.',
                        description: 'Source directory to use for the build. (Use \\for the path)',
                        name: 'BUILD_SRC_DIR')

        string(         defaultValue: params.BUILD_BLD_DIR ? params.BUILD_BLD_DIR : '.\\Build',
                        description: 'Build directory to use for the build. (Use \\ for the path)',
                        name: 'BUILD_BLD_DIR')

        //Discord
        credentials(    credentialType: 'org.jenkinsci.plugins.plaincredentials.impl.StringCredentialsImpl', 
                        defaultValue: params.DISCORD_WEBHOOK ? params.DISCORD_WEBHOOK : '', 
                        description: 'The webhook to use to notify the team for build results.', 
                        name: 'DISCORD_WEBHOOK', 
                        required: false)

    }
    
    agent {
        node {
            label ""
            customWorkspace "C:/Jenkins/${env.JOB_NAME}"
        }
    }
    
    options {
        timeout(time: 45, unit: 'MINUTES')
    }
    
    stages {

        stage('Pulling project') {
            //Pull project from GitHub.
            steps{
                echo "Pulling project ..."

                git branch: params.GITHUB_BRANCH, credentialsId: params.GITHUB_CREDENTIALS, url: params.GITHUB_URL

                echo "Finished pulling project."
            }
        }
        stage('Creating project'){
            steps{
                echo "Creating Project ..."

                bat "if not exist ${params.BUILD_BLD_DIR} " +
                    "(mkdir ${params.BUILD_BLD_DIR} && echo Created directory: ${params.BUILD_BLD_DIR}) " + 
                    "else (echo Already existing directory: ${params.BUILD_BLD_DIR})"
                bat "cmake -S ${params.BUILD_SRC_DIR} -B ${params.BUILD_BLD_DIR}"  

                echo "Finished creating project."
            }
            //Create visual studio project using Cmake.
        }
        stage('Compiling project') {
            steps{
                echo "Compiling project ..."

                bat "cmake --build ${params.BUILD_BLD_DIR}"

                echo "Finished compiling project."
            }
            //Compile visual studio project
        }
        stage('Testing project') {
            steps{
                echo "Testing project ..."
                script{
                    echo "Build name: ${currentBuild.displayName}"
                    echo "Build changes: ${currentBuild.changeSets}"
                    echo "Build causes: ${currentBuild.getBuildCauses()}"
                    echo "Env branch name: ${env.BRANCH_NAME}"
                    echo "Env change id: ${env.CHANGE_ID}"
                    echo "Env change url: ${env.CHANGE_URL}"
                    echo "Env change author: ${env.CHANGE_AUTHOR}"
                    echo "Env change author name: ${env.CHANGE_AUTHOR_DISPLAY_NAME}"
                    echo "Env change author email: ${env.CHANGE_AUTHOR_EMAIL}"
                    echo "Env change target: ${env.CHANGE_TARGET}"
                    echo "Env change branch: ${env.CHANGE_BRANCH}"
                    echo "Env change fork: ${env.CHANGE_FORK}" 
                }
                echo "Finished testing project."
            }
            //Run unit-tests, automated comparisons, etc.
        }
        stage('Publishing project') {
            steps{
                echo "Publishing project ..."
                echo "Finished publishing project."
            }
            //Publish project to public/stable GitHub repository.
        }
    }
    
    post {
        success {
            script {
                if(params.DISCORD_WEBHOOK) {
                    withCredentials([string(credentialsId: params.DISCORD_WEBHOOK, variable: 'WEBHOOK_URL')]) {
                        discord.sendEmbed(  env.WEBHOOK_URL, 
                                            "Ran Jenkins Pipeline for ${env.JOB_BASE_NAME}", 
                                            "Build #${env.BUILD_NUMBER}", 
                                            '3066993', 
                                            [['**Build Result**', ":white_check_mark: Build succeeded! \n :stopwatch: Build duration: ${currentBuild.durationString}"]],
                                            '');
                        echo "Build duration: ${currentBuild.durationString}"
                    }
                }
            }
        }
        
        unsuccessful {
            script {
                if(params.DISCORD_WEBHOOK) {
                    withCredentials([string(credentialsId: params.DISCORD_WEBHOOK, variable: 'WEBHOOK_URL')]) {
                        discord.sendEmbed(  env.WEBHOOK_URL, 
                                            "Ran Jenkins Pipeline for ${env.JOB_BASE_NAME}", 
                                            "Build #${env.BUILD_NUMBER}", 
                                            '15158332', 
                                            [['**Build Result**', ':x: Build failed!'],
                                            ['**Failed Stage**', 'stage??'],
                                            ['**Output**', '```Error output```']],
                                            '');
                    }
                }
            }
        }
 
        unstable {
            script {
                if(params.DISCORD_WEBHOOK) {
                    withCredentials([string(credentialsId: params.DISCORD_WEBHOOK, variable: 'WEBHOOK_URL')]) {
                        discord.sendEmbed(  env.WEBHOOK_URL, 
                                            "Ran Jenkins Pipeline for ${env.JOB_BASE_NAME}", 
                                            "Build #${env.BUILD_NUMBER}", 
                                            '16776960', 
                                            [['**Build Result**', ':warning: Build unstable!']],
                                            '');
                    }
                }
            }
        }
        
        always {
            script {
                cleanWs();
            }
        }
    }
}