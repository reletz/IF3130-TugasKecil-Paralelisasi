{
  description = "NixOS VM Cluster for OpenMPI";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: 
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      
      # Baca SSH public key dari host
      sshPublicKey = builtins.readFile "${builtins.getEnv "HOME"}/.ssh/id_ed25519.pub";

      clusterSshPublicKey = builtins.readFile ./mpi_cluster_key.pub;
      masterSshPrivateKeyFile = ./mpi_cluster_key;
      
      # Konfigurasi common untuk semua VM
      commonConfig = { config, pkgs, ... }: {
        # Networking
        networking.firewall.enable = false;
        networking.extraHosts = ''
          192.168.55.10 master
          192.168.55.11 worker1
          192.168.55.12 worker2
          192.168.55.13 worker3
        '';
        
        # User mpiuser
        users.users.mpiuser = {
          isNormalUser = true;
          extraGroups = [ "wheel" ];
          password = "mpipass";
          openssh.authorizedKeys.keys = [ sshPublicKey clusterSshPublicKey ];
        };
        
        # SSH
        services.openssh = {
          enable = true;
          settings.PermitRootLogin = "yes";
          settings.PasswordAuthentication = true;
        };
        
        # OpenMPI
        environment.systemPackages = with pkgs; [
          gcc
          openmpi
          openmpi.dev

          opencv
          pkg-config

          vim
          htop
        ];

        environment.variables = {
          PKG_CONFIG_PATH = "${pkgs.opencv}/lib/pkgconfig";
          OPENCV_INCLUDE = "${pkgs.opencv}/include";
          OPENCV_LIB = "${pkgs.opencv}/lib";
        };
        
        system.stateVersion = "24.05";
      };
      
      # Generate VM config
      makeVM = { name, ip, memory ? 1024, cores ? 2, extraModules ? [] }: nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          commonConfig
          {
            networking.hostName = name;
            networking.interfaces.eth1.ipv4.addresses = [ {
              address = ip;
              prefixLength = 24;
            } ];
            
            # Konfigurasi QEMU VM
            virtualisation.vmVariant = {
              virtualisation = {
                memorySize = memory;
                cores = cores;
                graphics = false;
                
                qemu.networkingOptions = [
                  "-net nic,netdev=user.1,model=virtio"
                  "-netdev user,id=user.1,hostfwd=tcp::${toString (10022 + (if name == "master" then 1 else if name == "worker1" then 2 else if name == "worker2" then 3 else 4))}-:22"

                  "-net nic,netdev=internal.2,model=virtio"
                  "-netdev socket,id=internal.2,mcast=230.0.0.1:1234"
                ];
              };
            };
          }
        ] ++extraModules;
      };
      
    in {
      # Define semua VM
      nixosConfigurations = {
        master = makeVM {
          name = "master";
          ip = "192.168.55.10";
          memory = 2048;
          cores = 4;
          extraModules = [
            ({ pkgs, ... }: {
              system.activationScripts.setupMpiEnvironment = {
                deps = [ "users" ];
                text = ''
                  echo "master  slots=4
                  worker1 slots=2
                  worker2 slots=2
                  worker3 slots=2" > /home/mpiuser/hostfile
                  chown mpiuser:users /home/mpiuser/hostfile
                '';
              };
            })
            ({ pkgs, ... }: {
              system.activationScripts.setupMasterSshKey = {
                deps = [ "users" ];
                text = ''
                  USER_SSH_DIR=/home/mpiuser/.ssh
                  mkdir -p $USER_SSH_DIR
                  cp ${masterSshPrivateKeyFile} $USER_SSH_DIR/id_ed25519
                  chown -R mpiuser:users $USER_SSH_DIR
                  chmod 700 $USER_SSH_DIR
                  chmod 600 $USER_SSH_DIR/id_ed25519
                '';
              };
            })
            ({ pkgs, ... }: {
              system.activationScripts.setupSshConfig = {
                deps = [ "users" ];
                text = ''
                  USER_SSH_DIR=/home/mpiuser/.ssh
                  mkdir -p $USER_SSH_DIR
                  
                  cat > $USER_SSH_DIR/config << 'EOF'
Host 192.168.55.* worker*
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
EOF

                  chown mpiuser:users $USER_SSH_DIR/config
                  chmod 600 $USER_SSH_DIR/config
                '';
              };
            })
          ];
        };
        
        worker1 = makeVM {
          name = "worker1";
          ip = "192.168.55.11";
          memory = 1024;
          cores = 2;
        };
        
        worker2 = makeVM {
          name = "worker2";
          ip = "192.168.55.12";
          memory = 1024;
          cores = 2;
        };
        
        worker3 = makeVM {
          name = "worker3";
          ip = "192.168.55.13";
          memory = 1024;
          cores = 2;
        };
      };
      
      # Packages untuk build VM
      packages.${system} = {
        master-vm = self.nixosConfigurations.master.config.system.build.vm;
        worker1-vm = self.nixosConfigurations.worker1.config.system.build.vm;
        worker2-vm = self.nixosConfigurations.worker2.config.system.build.vm;
        worker3-vm = self.nixosConfigurations.worker3.config.system.build.vm;

        cluster-vms = pkgs.symlinkJoin {
          name = "mpi-cluster";
          paths = [
            self.nixosConfigurations.master.config.system.build.vm
            self.nixosConfigurations.worker1.config.system.build.vm
            self.nixosConfigurations.worker2.config.system.build.vm
            self.nixosConfigurations.worker3.config.system.build.vm
          ];
        };
      };
    };
}