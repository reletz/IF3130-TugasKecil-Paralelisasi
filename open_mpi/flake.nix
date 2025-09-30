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
      
      # Konfigurasi common untuk semua VM
      commonConfig = { config, pkgs, ... }: {
        # Networking
        networking.firewall.enable = false;
        
        # User mpiuser
        users.users.mpiuser = {
          isNormalUser = true;
          extraGroups = [ "wheel" ];
          password = "mpipass";
          openssh.authorizedKeys.keys = [ sshPublicKey ];
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
            
            # Konfigurasi QEMU VM
            virtualisation.vmVariant = {
              virtualisation = {
                memorySize = memory;
                cores = cores;
                graphics = false;
                
                qemu.networkingOptions = [
                  "-net nic,netdev=user.1,model=virtio"
                  "-netdev user,id=user.1,hostfwd=tcp::${toString (10022 + (if name == "master" then 1 else if name == "worker1" then 2 else if name == "worker2" then 3 else 4))}-:22"
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
          ip = "10.0.2.15";
          memory = 2048;
          cores = 4;
          extraModules = [
            ({ pkgs, ... }: {
              system.activationScripts.setupMpiEnvironment = {
                deps = [ "users" ];
                text = ''
                  echo "10.0.2.15
                  10.0.2.16
                  10.0.2.17
                  10.0.2.18" > $MPI_USER_HOME/home/mpiuser/hostfile
                  chown mpiuser:users $MPI_USER_HOME/home/mpiuser/hostfile
                '';
              };
            })
          ];
        };
        
        worker1 = makeVM {
          name = "worker1";
          ip = "10.0.2.16";
          memory = 1024;
          cores = 2;
        };
        
        worker2 = makeVM {
          name = "worker2";
          ip = "10.0.2.17";
          memory = 1024;
          cores = 2;
        };
        
        worker3 = makeVM {
          name = "worker3";
          ip = "10.0.2.18";
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